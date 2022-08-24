# -*- coding: utf-8 -*-
import logging

from fl_aggregator_libs import *
from random import Random
from resource_manager import ResourceManager
#Ahmed imported modules
import wandb
import torch
import copy
import traceback
from utils.utils_model import cosine_sim, kl_divergence, normalize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt




class Aggregator(object):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.device = torch.device('cpu') #args.cuda_device if args.use_cuda else torch.device('cpu')
        self.executors = [int(v) for v in str(args.learners).split('-')]
        self.num_executors = len(self.executors)

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.virtual_client_clock = {}
        self.round_duration = 0.
        self.resource_manager = ResourceManager()
        self.client_manager = self.init_client_manager(args=args)




        # ======== model and data ========
        self.model = None

        # list of parameters in model.parameters()
        self.model_in_update = []
        self.last_global_model = []

        # ======== channels ========
        self.server_event_queue = {}
        self.client_event_queue = Queue()
        self.control_manager = None
        # event queue of its own functions
        self.event_queue = collections.deque()

        # ======== runtime information ========
        self.tasks_round = 0
        self.sampled_participants = []

        self.round_stragglers = []
        self.roundclients=[]
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.epoch = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # Arouj - metrics for EAFL
        self.stats_energy_accumulator = []
        self.stats_energyeff_accumulator = []
        self.dropoutdueto_battery=[]




        # number of registered executors
        self.registered_executor_info = 0
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

        self.gradient_controller = None
        if self.args.gradient_policy == 'yogi':
            from utils.yogi import YoGi
            self.gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

        # Ahmed - metrics, stale and availibility variables
        self.clients_select_count = {}
        self.clients_success_count = {}
        self.clients_fail_count = {}
        # Ahmed - define the stale updates list
        self.staleWeights = {}
        self.staleRemainDuration = {}
        self.stale_rounds = {}
        self.round_stale_updates = 0
        self.straggler_comp_time = {}
        self.strugglers_to_run = []
        self.dropout_clients = []


        self.last_round_comp_time = {}
        self.last_round_stragglers = []
        self.last_round_duration = 0

        self.round_failures = 0
        self.mov_avg_deadline = 0
        self.deadline = 0
        self.attended_clients = 0
        self.unique_attend_clients = []
        self.acc_accumulator = []
        self.acc_5_accumulator = []
        self.train_accumulator = []
        self.completion_accumulator = []

        self.total_compute = 0
        self.total_communicate = 0

        # Arouj - used for tracking energy consumption
        self.total_energy=0
        self.total_energyeff = 0

        # Ahmed - used for choosing a replacement client among online if it matches the stale one
        self.rng = random.Random()

        self.total_updates = 0
        self.unused_stale = 0
        self.round_update = False
        # ======== Task specific ============
        self.imdb = None           # object detection

    def setup_env(self):
        self.setup_seed(seed=self.this_rank)

        # set up device
        # if self.args.use_cuda:
        #     if self.device == None:
        #         for i in range(torch.cuda.device_count()):
        #             try:
        #                 self.device = torch.device('cuda:'+str(i))
        #                 torch.cuda.set_device(i)
        #                 _ = torch.rand(1).to(device=self.device)
        #                 logging.info(f'End up with cuda device ({self.device})')
        #                 break
        #             except Exception as e:
        #                 assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'
        #     else:
        #         torch.cuda.set_device(self.device)
        logging.info(f'=== PS cuda device is preset to ({self.device})')

        #Ahmed - setup the seed again
        self.setup_seed(seed=self.this_rank)

        self.init_control_communication(self.args.ps_ip, self.args.manager_port, self.executors)
        self.init_data_communication()

    # Ahmed - for reporducbility - it does not work ---(okay)
    # Ahmed - https://github.com/pytorch/pytorch/issues/7068
    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        #torch.use_deterministic_algorithms(True)

    def init_control_communication(self, ps_ip, ps_port, executors):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Start to initiate {ps_ip}:{ps_port} for control plane communication ...")

        dummy_que = {executorId:Queue() for executorId in executors}
        # create multiple queue for each aggregator_executor pair
        for executorId in executors:
            BaseManager.register('get_server_event_que'+str(executorId), callable=lambda: dummy_que[executorId])

        dummy_client_que = Queue()
        BaseManager.register('get_client_event', callable=lambda: dummy_client_que)

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        self.control_manager.start()

        #self.server_event_queue = torch.multiprocessing.Manager().dict()
        for executorId in self.executors:
            self.server_event_queue[executorId] = eval('self.control_manager.get_server_event_que'+str(executorId)+'()')

        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)


    def init_model(self):
        """Load model"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

        return init_model()

    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Kuiper sampler
                - Kuiper prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://arxiv.org/abs/2010.06081
        """

        # sample_mode: random or kuiper
        client_manager = clientManager(args.sample_mode, args=args)
        return client_manager

    def load_client_profile(self, file_path):
        # load client profiles
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def sel_category(self, mapped_id):
        """Select category based on computational latency in the trace pickle file"""
        if self.args.device_conf_file is not None:
            with open(self.args.device_conf_file, 'rb') as fiz:
                a = pickle.load(fiz)
        x = []
        y = []
        div = []
        mid = []
        low = []
        high = []

        for key, val in a.items():
            if (key == mapped_id):
                if (val['computation'] < 68 and val['computation'] > 0):
                    category = 'H'
                if (val['computation'] < 132) and val['computation'] > 68:
                    category = 'M'
                if (val['computation'] < 200 and val['computation'] > 132):
                    category = 'L'
        return category

    def executor_info_handler(self, executorId, info):
        #logging.info(info)
        self.registered_executor_info += 1

        # have collected all executors
        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout

        if self.registered_executor_info == self.num_executors:

            clientId = 1


            for index, _size in enumerate(info['size']):
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = clientId%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

                # Arouj- We categorize each client into high, mid or low-end device
                category = self.sel_category(mapped_id)

                self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile, category= category)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                upload_epoch=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
                # Arouj - set initial battery level of each client
                self.client_manager.setbatterylevel(clientId, category=category)

                # Ahmed - initiate the client selection and run metrics
                self.clients_select_count[clientId] = 0
                self.clients_success_count[clientId] = 0
                self.clients_fail_count[clientId] = 0
                clientId=clientId+1

            logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))

            # start to sample clients
            self.round_completion_handler()


    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """We try to remove dummy events as much as possible, by removing the stragglers/offline clients in overcommitment"""

        sampledClientsReal = []
        dropoutClients = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            #logging.info(f'{client_to_run} {self.client_conf}')
            client_cfg = self.client_conf.get(client_to_run, self.args)

            exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                    batch_size=client_cfg.batch_size, upload_epoch=client_cfg.local_steps,
                                    upload_size=self.model_update_size, download_size=self.model_update_size)

            roundDuration = exe_cost['computation'] + exe_cost['communication']

            # Ahmed - account for the computation, communication and energy

            self.total_compute += exe_cost['computation']
            self.total_communicate += exe_cost['communication']


            # if the client has less battery level by the time of collection, we consider it as a drop-out
            if self.client_manager.isClientAlive(client_to_run, roundDuration):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = roundDuration               #exe_cost
            else:
                if(client_to_run not in dropoutClients):
                    dropoutClients.append(client_to_run)
                if (client_to_run not in self.dropoutdueto_battery):
                    self.dropoutdueto_battery.append(client_to_run)



        #Ahmed - if we do not filter dropouts, we inflate their completion timeu
        if args.no_filter_dropouts > 0:
            for client_to_run in dropoutClients:
                comp_time = args.no_filter_dropouts * roundDuration  # exe_cost
                # comp_time = args.no_filter_dropouts * 60 # no_filter_dropouts is # of minutes to wait for dropout
                sampledClientsReal.append(client_to_run)
                completionTimes.append(comp_time)
                completed_client_clock[client_to_run] = comp_time

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))

        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])

        # Ahmed - Change to only top 80% of the clients to simulate failure of 20% with deadline
        #if args.deadline_percent > 0:
         ##   num_clients_to_collect = int(math.floor(args.deadline_percent * num_clients_to_collect))

        # Ahmed - apply fixed deadline or moving average if it is not zero
        #deadline = args.deadline
        #if deadline < 0:
         #   if self.epoch > 1:
          #      id = sortedWorkersByCompletion[num_clients_to_collect - 1]
           #     if deadline == -2:
            #        id = sortedWorkersByCompletion[math.ceil(num_clients_to_collect * args.target_ratio) - 1]
             #   cid = sampledClientsReal[id]
              #  self.mov_avg_deadline =  (1 - args.deadline_alpha) * completed_client_clock[cid] + args.deadline_alpha * self.mov_avg_deadline
              #  deadline = self.mov_avg_deadline
            #else:
             #   deadline = self.args.initial_deadline
        #if deadline > 0:
         #   for index, i in enumerate(sortedWorkersByCompletion):
          #      if completionTimes[i] > deadline:
           #         num_clients_to_collect = index
            #        break

        top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
        #logging.info("====Apply deadline {}:{}:{} or percent {}, before num {} final num {} sorted workers {} topk index {} durations: {}".format(
         #       args.deadline, args.target_ratio, deadline, args.deadline_percent, len(sortedWorkersByCompletion), len(top_k_index),
          #      sortedWorkersByCompletion, top_k_index, completed_client_clock))
        clients_to_run = []
        dummy_clients = []
        round_duration = 0.0
        if len(top_k_index):
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]
            #dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:] if sampledClientsReal[k] not in dropoutClients]
            round_duration = completionTimes[top_k_index[-1]]

        return clients_to_run, dummy_clients, dropoutClients, completed_client_clock, round_duration, completionTimes

    def run(self):
        try:
            self.setup_env()
            self.model = self.init_model()

            #Ahmed - get the param count
            self.model_param_count = 0
            for idx, param in enumerate(self.model.parameters()):
                self.model_param_count += 1

            self.save_last_param()

            self.model_update_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
            self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)
            self.start_event()
            self.event_monitor()
        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # Ahmed - add print of the stack call trace to give meaningful debugging information
            logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
            print('Aggregator Exception - sending stop signal and terminating the process group Now!')
            self.broadcast_msg('emergency_stop')
            time.sleep(5)
            self.stop()
            exit(0)

    def start_event(self):
        #self.model_in_update = [param.data for idx, param in enumerate(self.model.parameters())]
        #self.event_queue.append('report_executor_info')
        pass

    def broadcast_msg(self, msg):
        for executorId in self.executors:
            self.server_event_queue[executorId].put_nowait(msg)


    def broadcast_models(self):
        """Push the latest model to executors"""
        # self.model = self.model.to(device='cpu')

        # waiting_list = []
        for param in self.model.parameters():
            temp_tensor = param.data.to(device='cpu')
            for executorId in self.executors:
                dist.send(tensor=temp_tensor, dst=executorId)
                # req = dist.isend(tensor=param.data, dst=executorId)
                # waiting_list.append(req)

        # for req in waiting_list:
        #     req.wait()

        # self.model = self.model.to(device=self.device)


    def select_participants(self, select_num_participants, overcommitment=1.3, time_window=0):
        #Ahmed - change to include the deadline for availability calculations
        #return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock))
        return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock, time_window=time_window))


    def client_completion_handler(self, results, importance=1.0):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""

        clientId=results['clientId']

        complete_time = 0
        if clientId in self.virtual_client_clock:
            complete_time = self.virtual_client_clock[clientId] #self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication']
            client_cfg = self.client_conf.get(clientId, self.args)
            exe_cost = self.client_manager.getCompletionTime(clientId,
                                                             batch_size=client_cfg.batch_size,
                                                             upload_epoch=client_cfg.local_steps,
                                                             upload_size=self.model_update_size,
                                                             download_size=self.model_update_size)
            # Arouj - Keep track of energy metrics
            energy = self.client_manager.getEnergy(clientId, exe_cost['computation'], exe_cost['communication'])
            updateutil= self.client_manager.updateutility(clientId, energy)
            updateutil *= 3600
            energyeff = self.client_manager.getenergyeff(clientId, energy=energy, batch_size=client_cfg.batch_size)

            # Arouj -Updated reward function
            upd_util = ((results['utility'] * args.scal_val) + (1-args.scal_val) * updateutil)
            results['utility'] = upd_util

        #Ahmed - record the number of attended and unique clients
        if clientId not in self.unique_attend_clients:
            self.unique_attend_clients.append(clientId)
        self.attended_clients += 1

        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        self.client_training_results.append(results)
        self.stats_energy_accumulator.append(energy)
        self.stats_energyeff_accumulator.append(energyeff)

        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        #Ahmed - accumlate train_acc and train_acc_5
        self.acc_accumulator.append(results['train_acc'] * 100)
        self.acc_5_accumulator.append(results['train_acc_5'] * 100)
        self.train_accumulator.append(results['train_loss'])
        self.completion_accumulator.append(complete_time)

        # Arouj - accumulate energy and energyeff
        self.total_energy += energy
        self.total_energyeff += energyeff

        #Ahmed - Handle and keep a copy of the stale client update
        isStale = True if (clientId in self.staleWeights and len(self.staleWeights[clientId]) == 0) else False
        if isStale:
            self.staleWeights[clientId] = copy.deepcopy(results['update_weight'])
            #complete_time = self.straggler_comp_time[clientId] * args.straggler_penalty
            logging.info('======== Aggregator received straggler client: id {} comp_time {} param_len {} stale {}'.format(clientId, complete_time, len(self.staleWeights[clientId]), len(self.staleWeights)))
        else:
           # Energyeff = self.client_manager.getEnergyeff(results['clientId'], 100, 100)
            self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
                    time_stamp=self.epoch, duration=complete_time)

        #Ahmed - only update the model if the round did not fail (we have enough new+stale clients)
        # N1 = list()
        # for vector in results['update_weight']:
        #     N1.append(np.linalg.norm(vector, ord=None))
        if self.round_update and not isStale:
            # Start to take the average of updates, and we do not keep updates to save memory
            # Importance of each update is 1/#_of_participants
            #importance = 1. / self.tasks_round
            if len(self.model_in_update) == 0:
                self.model_in_update = [True]
                for idx, param in enumerate(self.model.parameters()):
                    param.data = torch.from_numpy(results['update_weight'][idx]).to(device=self.device) * importance
            else:
                for idx, param in enumerate(self.model.parameters()):
                    param.data += torch.from_numpy(results['update_weight'][idx]).to(device=self.device) * importance
            #logging.info(f"====== Aggregator UPDATE model using {clientId} update with importance {importance}: norm {sum(N1)}")
        #elif self.round_update:
            #logging.info(f"====== Aggregator STORE {clientId} struggler update with importance {importance}: norm {sum(N1)}")

    def save_last_param(self):
        self.last_global_model = [param.data.clone() for param in self.model.parameters()]

    def round_weight_handler(self, last_model, current_model):
        if self.epoch > 1:
            if self.args.gradient_policy == 'yogi':
                last_model = [x.to(device=self.device) for x in last_model]
                current_model = [x.to(device=self.device) for x in current_model]

                diff_weight = self.gradient_controller.update([pb-pa for pa, pb in zip(last_model, current_model)])

                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] + diff_weight[idx]

            elif self.args.gradient_policy == 'qfedavg':

                learning_rate, qfedq = self.args.learning_rate, self.args.qfed_q
                Deltas, hs = None, 0.
                last_model = [x.to(device=self.device) for x in last_model]

                for result in self.client_training_results:
                    # plug in the weight updates into the gradient
                    grads = [(u - torch.from_numpy(v).to(device=self.device)) * 1.0 / learning_rate for u, v in zip(last_model, result['update_weight'])]
                    loss = result['moving_loss']

                    if Deltas is None:
                        Deltas = [np.float_power(loss+1e-10, qfedq) * grad for grad in grads]
                    else:
                        for idx in range(len(Deltas)):
                            Deltas[idx] += np.float_power(loss+1e-10, qfedq) * grads[idx]

                    # estimation of the local Lipchitz constant
                    hs += (qfedq * np.float_power(loss+1e-10, (qfedq-1)) * torch.sum(torch.stack([torch.square(grad).sum() for grad in grads])) + (1.0/learning_rate) * np.float_power(loss+1e-10, qfedq))

                # update global model
                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] - Deltas[idx]/(hs+1e-10)

    def round_completion_handler(self):
        # update the virtual clock
        # Ahmed - change to use the deadline if set
        duration = self.round_duration
        if self.deadline != 0:
            duration = self.deadline
        self.global_virtual_clock += duration


        # reduction of battery for unselected clients for this round
        self.client_manager.idlepowerdeduction( self.roundclients, duration)


        # Ahmed - clock update
        # if self.deadline == 0 or self.tasks_round >= args.total_worker and args.last_worker:
        #     self.global_virtual_clock += self.round_duration
        # else:
        #     self.global_virtual_clock += self.deadline

        self.epoch += 1

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        #Ahmed - change to apply weight updates if round not failed
        if self.round_update:
            # handle the global update w/ current and last
            self.round_weight_handler(self.last_global_model, [param.data.clone() for param in self.model.parameters()])

        avgUtilLastEpoch = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))

        if args.stale_update <= 0:
            # assign avg reward to explored, but not ran workers
            for clientId in self.round_stragglers:
                complete_time=self.virtual_client_clock[clientId] #self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication']
              #  energyeff =  self.client_manager.getEnergyeff (clientId, complete_time['computation'] , complete_time['communication'])
                self.client_manager.registerScore(clientId, avgUtilLastEpoch, time_stamp=self.epoch, duration= complete_time,success=False)

                
        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, Epoch: {self.epoch}, Planned participants: " + \
            f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # update select participants
        # Ahmed - use the version with availability prioritization
      #  self.deadline = args.deadline
       # overcommitment = 1.0
        #if self.deadline <= 0:
         #   if self.mov_avg_deadline == 0:
          #      # Ahmed - set the initial deadline to something large
           #     self.deadline = self.args.initial_deadline
           # else:
            #    self.deadline = self.mov_avg_deadline
             #   self.cliedeadline = self.mov_avg_deadline
        #Ahmed - original oort setting
       # if args.exp_type == 1 or args.exp_type == 3:
        #    overcommitment = self.args.overcommitment
         #   self.deadline = self.round_duration
       # time_window = self.deadline if self.deadline > 0 else self.round_duration

        self.sampled_participants = self.select_participants(select_num_participants=self.args.total_worker, overcommitment=self.args.overcommitment)
        #self.sampled_participants = self.select_participants(select_num_participants=self.args.total_worker, overcommitment=overcommitment)
        #logging.info(self.sampled_participants)

        #Ahmed - count the number of times a client is selected
        for c in self.sampled_participants:
            self.clients_select_count[c] += 1

        #Ahmed - changed the way clients are filtered using deadline or deadline_percent
        
        clientsToRun, round_stragglers, dropout_clients, virtual_client_clock, round_duration, client_completions = self.tictak_client_tasks(self.sampled_participants, self.args.total_worker)

            # Ahmed - filter out the clients with pending stale updates if they were selected again
        # Ahmed - filter out the clients with pending stale updates if they were selected again
        if self.args.stale_update:
            count = 0
            sel_stale = []
            replacement_clients = []
            for c in clientsToRun:
                if c in self.staleWeights:
                    count += 1
                    sel_stale.append(c)
                    clientsToRun.remove(c)
            sel_stale_temp = sel_stale
            while len(sel_stale) > 0:
                c = sel_stale.pop()
                #online_clients, _ = self.client_manager.getOnlineClients(cur_time=self.global_virtual_clock)
                index = self.rng.randint(0, len(self.client_manager.cur_alive_clients) - 1)
                while self.client_manager.cur_alive_clients[index] == c:
                    index = self.rng.randint(0, len(self.client_manager.cur_alive_clients) - 1)
                replacement_clients.append(self.client_manager.cur_alive_clients[index])
                clientsToRun.append(self.client_manager.cur_alive_clients[index])
            logging.info('Resample clients: round {} num {} stale {} stale_clients {} replacement {} sampled {}'.format(\
                         self.epoch, self.args.total_worker, count, sel_stale_temp, replacement_clients, clientsToRun))

        # Ahmed - round's successful clients
        self.tasks_round = len(clientsToRun)
        for c in clientsToRun:
            self.clients_success_count[c] += 1
        for c in round_stragglers:
            self.clients_fail_count[c] += 1

        # Ahmed - get the number of stale clients finishing this round
        self.round_stale_updates, self.unused_stale = self.get_stale_status()
        self.total_updates = self.tasks_round + self.round_stale_updates

        #Ahmed - account for drop out clients if no_filter_dropouts is set
        if self.args.no_filter_dropouts:
            self.total_updates -= len(dropout_clients)

        # Ahmed - Perform round update only if the target ratio is met or target ratio is 0 (accept all)
        self.round_update = True

        # enforce round failures if experiment type is 2 (deadline with failure) or 3 (overcommit with failure)
        #Ahmed - exp_type = 2 is deadline with target_ratio, if target number is not met, round fails
        #Ahmed - exp_type= 3 is overcommit with 100% target of the total_workers but fails if dropouts exceeds overcommit
        if (args.exp_type == 2 and self.total_updates < math.floor(args.total_worker * args.target_ratio)) \
                     or (args.exp_type == 3 and self.total_updates < args.total_worker):
            self.round_update = False
            self.round_failures += 1

        # Ahmed - extend the clients to run with the strugglers (stale clients) if they are not in the stale state
        self.strugglers_to_run = []
        if args.stale_update > 0:
            for clientId in round_stragglers:
                if clientId not in self.staleWeights:
                    self.staleWeights[clientId] = []
                    self.stale_rounds[clientId] = 0
                    self.staleRemainDuration[clientId] = virtual_client_clock[clientId]
                    self.straggler_comp_time[clientId] = virtual_client_clock[clientId]
                    self.strugglers_to_run.append(clientId)
            clientsToRun.extend(self.strugglers_to_run)
        logging.info(f"Selected participants to run len {self.round_update}:{self.total_updates}:{len(clientsToRun)}:{self.tasks_round}:{len(self.strugglers_to_run)}:{len(round_stragglers)}:{len(virtual_client_clock)}: {clientsToRun}:\n{virtual_client_clock}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.save_last_param()

        #Collect info from last round
        self.last_round_stragglers = self.round_stragglers
        self.last_round_comp_time = self.virtual_client_clock
        self.last_round_duration = self.round_duration

        self.roundclients=clientsToRun
        self.round_stragglers = round_stragglers
        self.dropout_clients = dropout_clients
        self.virtual_client_clock = virtual_client_clock
        self.round_duration = round_duration

        if self.epoch >= self.args.epochs:
            self.event_queue.append('stop')

        elif len(clientsToRun) <= 0:
            #Ahmed - handle the case when we have 0 clients to run
            self.event_queue.append('skip_round')

        elif self.epoch % self.args.eval_interval == 0:
            self.event_queue.append('update_model')
            self.event_queue.append('test')
        else:
            self.event_queue.append('update_mo'
                                    'del')
            self.event_queue.append('start_round')

    #Ahmed - define function resetting round metrics
    def round_reset_metrics(self):
        self.model_in_update = []
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        # Ahmed - reset accumulator metrics
        self.acc_5_accumulator = []
        self.acc_accumulator = []
        self.train_accumulator = []
        self.completion_accumulator = []

        self.unused_stale = 0

    def testing_completion_handler(self, results):
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                    }
            else:
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'test_len': accumulator['test_len']
                    }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                    self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['loss'],
                    self.testing_history['perf'][self.epoch]['test_len']))

            #Ahmed - wandb logs
            if args.use_wandb :
                # Ahmed - log the test performance
                wandb.log({'Test/acc_top_1': self.testing_history['perf'][self.epoch]['top_1'],
                           'Test/acc_top_5': self.testing_history['perf'][self.epoch]['top_5'],
                           'Test/loss': self.testing_history['perf'][self.epoch]['loss'],
                           #'Test/data_len': self.testing_history['perf'][self.epoch]['test_len'],
                           }, step=self.epoch)

                #Ahmed - add more metrics on top 5 test accuracy
                if 'test' in self.testing_history['perf'][self.epoch]:
                    test_accs = np.asarray(self.testing_history['perf'][self.epoch]['test'])
                    num_test_clients = len(test_accs)
                    wandb.log({"Test/top_5_avg": np.average(test_accs, axis=0),
                               "Test/top_5_10p": np.percentile(test_accs, 10, axis=0),
                               "Test/top_5_50p": np.percentile(test_accs, 50, axis=0),
                               "Test/top_5_90p": np.percentile(test_accs, 90, axis=0),
                               "Test/top_5_var": np.var(test_accs),
                               "Round/test_clients": num_test_clients,
                               }, step=self.epoch)

                    if  num_test_clients > 0:
                        wandb.log({"Clients/test_top_5": wandb.Histogram(np_histogram=np.histogram(test_accs, bins=10)),}, step=self.epoch)

                        # Ahmed - Fairness of test accuracy
                        wandb.log({"Fairness/jain_top_5": (1.0 / num_test_clients * (np.sum(test_accs) ** 2) / np.sum(test_accs ** 2)),
                                   "Fairness/qoe_top_5": (1.0 - (2.0 * test_accs.std() / (test_accs.max() - test_accs.min()))),
                                   }, step=self.epoch)

                        # Compute an log cosine similarity metric with input vector a of
                        # clients' accuracies and b the same-length vector of 1s
                        vectors_of_ones = np.ones(num_test_clients)
                        wandb.log({"Fairness/cs_test_top_5":cosine_sim(test_accs, vectors_of_ones)}, step=self.epoch)

                        # Compute an log KL Divergence metric with input vector a of
                        # clients' normalized accuracies and the vector b of same-length
                        # generated from the uniform distribution
                        uniform_vector = np.random.uniform(0, 1, num_test_clients)
                        wandb.log({"Fairness/kl_test_top_5":kl_divergence(normalize(test_accs), uniform_vector)
                        }, step=self.epoch)

            self.event_queue.append('start_round')

    #Ahmed - define stale clients handler
    def get_stale_status(self):
        stale_count = 0
        unused_stale = 0
        if len(self.staleWeights) > 0:
            for clientId in list(self.staleWeights):
                if (len(self.staleWeights[clientId]) > 0 and clientId in self.staleRemainDuration and self.staleRemainDuration[clientId] < self.deadline):
                    if self.stale_rounds[clientId] <= args.stale_update:
                        stale_count += 1
                    else:
                        unused_stale += 1
        return stale_count, unused_stale

    def stale_clients_handler(self, importance=1.0):
        # Ahmed - update the remaining time or delete the stale update if updated already
        if len(self.staleWeights) > 0:
            stale_updates = []
            for clientId in list(self.staleWeights):
                stale_client_out = False
                # if self.stale_rounds[clientId] > args.stale_update:
                #     logging.info("==== Stale client {} exceed allowed staleness rounds: len {} remaining {} round duration {} stale rounds {} max stale {}".format(
                #             clientId, len(self.staleWeights[clientId]), self.staleRemainDuration[clientId], self.deadline,
                #             self.stale_rounds[clientId], args.stale_update))
                #     stale_client_out = True
                if  len(self.staleWeights[clientId]) > 0 and clientId in self.staleRemainDuration:
                    if self.staleRemainDuration[clientId] <= self.deadline:
                        if self.stale_rounds[clientId] <= args.stale_update:
                            logging.info("==== Stale client {} param update: len {} remaining {} round duration {} stale rounds {} max stale {}".format(
                                    clientId, len(self.staleWeights[clientId]), self.staleRemainDuration[clientId],  self.deadline, self.stale_rounds[clientId], args.stale_update))
                            stale_updates.append(self.staleWeights[clientId])
                        else:
                            logging.info("==== Stale client {} exceed allowed staleness rounds: len {} remaining {} round duration {} stale rounds {} max stale {}".format(clientId, len(self.staleWeights[clientId]), self.staleRemainDuration[clientId], self.deadline, self.stale_rounds[clientId], args.stale_update))
                        stale_client_out = True
                    else:
                        logging.info("==== Stale client {} stay in cache: len {} remaining {} round duration {} stale rounds {} max stale {}".format(
                                clientId, len(self.staleWeights[clientId]), self.staleRemainDuration[clientId],  self.deadline,
                                self.stale_rounds[clientId], args.stale_update))
                        self.staleRemainDuration[clientId] -= self.deadline
                        self.stale_rounds[clientId] += 1
                if stale_client_out == True:
                    del self.staleWeights[clientId]
                    del self.staleRemainDuration[clientId]
                    del self.stale_rounds[clientId]

            num_stale_updates = len(stale_updates)
            if num_stale_updates != self.round_stale_updates:
                logging.info(f'===== AGREGATOR LOGICAL ERROR {num_stale_updates} vs {self.round_stale_updates}')
            if self.round_update and num_stale_updates > 0:
                start = 0
                if len(self.model_in_update) == 0:
                    for idx, param in enumerate(self.model.parameters()):
                        param.data = torch.from_numpy(stale_updates[0][idx]).to(device=self.device) * importance
                    self.model_in_update = [True]
                    start = 1
                for idx, param in enumerate(self.model.parameters()):
                    for i in range(start, len(stale_updates)):
                        param.data += torch.from_numpy(stale_updates[i][idx]).to(device=self.device) * importance
            # for i, update in enumerate(stale_updates):
            #     N1 = list()
            #     for vector in update:
            #         N1.append(np.linalg.norm(vector, ord=None))
            #     logging.info(f"====== Aggregator STALE {i} importance {importance}: norm {sum(N1)}")
            #logging.info(f'====== Aggregator updated the model using {num_stale_updates} stale updates with importance {importance}' )
            del stale_updates
            gc.collect()

    def log_round_metrics(self):
        # Ahmed - process values
        # Ahmed - log the train metrics information to file and wandb
        # rewards_dict = {x: clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}
        # rewards_list = list(rewards_dict.values())
        #rem_durations = [x-self.round_duration for x  in list(self.staleRemainDuration.values())]
        rem_durations = [max(0, v) for k,v in list(self.staleRemainDuration.items()) if self.stale_rounds[k] > 0]
        stale_rounds_list = [v for v in list(self.stale_rounds.values()) if v > 0]

        #calculate model norm
        N1 = list()
        for idx, param in enumerate(self.model.parameters()):
            N1.append(np.linalg.norm(param.detach().cpu().numpy(), ord=None))
        agg_norm = sum(N1)

        # Ahmed - log scalars to wandb
        if args.use_wandb and self.epoch > 1:
            wandb.log({'Round/selected_clients': len(self.sampled_participants),
                       'Round/success_clients': len(self.stats_util_accumulator) if self.stats_util_accumulator else 0,
                       'Round/energy_clients': sum(self.stats_energy_accumulator) if self.stats_energy_accumulator else 0,
                       'Round/energyeff_clients': sum(self.stats_energyeff_accumulator) if self.stats_energyeff_accumulator else 0,
                       'Round/failed_clients': len(self.round_stragglers),
                       'Round/dropout_clients': len(self.dropout_clients),
                       'Round/dropout_battery': len(self.dropoutdueto_battery),
                        'Round/stale_updates': len(self.staleWeights),
                       'Round/attended_clients': self.attended_clients,
                       'Round/unique_clients': len(self.unique_attend_clients),
                       'Round/alive_clients': len(self.client_manager.cur_alive_clients),
                       'Round/clock': self.global_virtual_clock,
                       'Round/epoch': self.epoch,
                       'Round/duration': self.round_duration,
                       'Round/mov_avg_deadline': self.mov_avg_deadline,
                       'Round/deadline': self.args.deadline if self.args.deadline > 0 else self.mov_avg_deadline,
                       'Round/compute': self.total_compute,
                       'Round/energy': self.total_energy,
                       'Round/energyeff': self.total_energyeff,
                       'Round/communicate': self.total_communicate,
                       'Round/unused_stale': self.unused_stale,
                       'Round/param_norm': agg_norm,



                       #'Train/avg_loss': np.average(self.loss_accumulator),
                       'Train/acc_top_1': np.average(self.acc_accumulator),
                       'Train/acc_top_5': np.average(self.acc_5_accumulator),
                       'Train/loss': np.average(self.train_accumulator),

                       'Clients/avg_reward': np.average(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/std_reward': np.std(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/min_reward': np.min(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/max_reward': np.max(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/avg_energyeff': np.average(self.stats_energyeff_accumulator) if len(
                           self.stats_energyeff_accumulator) > 0 else 0,
                       'Clients/min_energyeff': np.min(self.stats_energyeff_accumulator) if len(
                           self.stats_energyeff_accumulator) > 0 else 0,
                       'Clients/max_energyeff': np.max(self.stats_energyeff_accumulator) if len(
                           self.stats_energyeff_accumulator) > 0 else 0,
                       'Clients/avg_energy': np.average(self.stats_energy_accumulator) if len(
                           self.stats_energy_accumulator) > 0 else 0,
                       'Clients/min_energy': np.min(self.stats_energy_accumulator) if len(
                           self.stats_energy_accumulator) > 0 else 0,
                       'Clients/max_energy': np.max(self.stats_energy_accumulator) if len(
                           self.stats_energy_accumulator) > 0 else 0,
                       'Clients/acc_energy': self.stats_energy_accumulator if len(
                           self.stats_energy_accumulator) > 0 else 0,




                       'Clients/avg_compute': np.average(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/std_compute': np.std(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/min_compute': np.min(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/max_compute': np.max(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,

                       'Stale/num_clients': len(self.staleWeights),
                       'Stale/avg_rem_duration': np.average(rem_durations) if len(rem_durations) else 0,
                       'Stale/std_rm_durations': np.std(rem_durations) if len(rem_durations) else 0,
                       'Stale/avg_rounds': np.average(stale_rounds_list) if len(stale_rounds_list) else 0,
                       'Stale/max_rounds': np.max(stale_rounds_list) if len(stale_rounds_list) else 0,
                       'Stale/min_rounds': np.min(stale_rounds_list) if len(stale_rounds_list) else 0,
                       }, step=self.epoch)

            #Ahmed - log the fairness of the selection process
            clients_selected = np.asarray(list(self.clients_select_count.values()))
            clients_success = np.asarray(list(self.clients_success_count.values()))
            clients_fail = np.asarray(list(self.clients_fail_count.values()))
            wandb.log({"Fairness/jain_selection": (1.0 / len(clients_selected) * (np.sum(clients_selected) ** 2) / np.sum(clients_selected ** 2)),
                       "Fairness/qoe_selection": (1.0 - (2.0 * clients_selected.std() / (clients_selected.max() - clients_selected.min()))),
                       "Fairness/jain_success": (1.0 / len(clients_success) * (np.sum(clients_success) ** 2) / np.sum(clients_success ** 2)),
                       "Fairness/qoe_success": (1.0 - (2.0 * clients_success.std() / (clients_success.max() - clients_success.min()))),
                       "Fairness/jain_fail": (1.0 / len(clients_fail) * (np.sum(clients_fail) ** 2) / np.sum(clients_fail ** 2)),
                        "Fairness/qoe_fail": (1.0 - (2.0 * clients_fail.std() / (clients_fail.max() - clients_fail.min()))),
                       }, step=self.epoch)


            #Ahmed - log stale updates
            # log updates from successful clients and stale updates applied in this round
            wandb.log({'Round/new_updates': self.tasks_round ,
                       'Round/stale_updates': self.round_stale_updates,
                       'Round/round_failures': self.round_failures,
                       'Round/total_updates': self.total_updates},
                      step=self.epoch)

            # Ahmed - log histograms to wandb
            wandb.log({"Clients/train_acc": wandb.Histogram(np_histogram=np.histogram(self.acc_accumulator, bins=10)),
                       "Clients/train_acc_5": wandb.Histogram(np_histogram=np.histogram(self.acc_5_accumulator, bins=10)),
                       "Clients/train_loss": wandb.Histogram(np_histogram=np.histogram(self.train_accumulator, bins=10)),
                       "Clients/rewards": wandb.Histogram(np_histogram=np.histogram(self.stats_util_accumulator, bins=10)),
                       #"Clients/PPW": wandb.Histogram(np_histogram=np.histogram(self.stats_energyeff_accumulator, bins=10)),
                       "Clients/completion_time": wandb.Histogram(np_histogram=np.histogram(self.completion_accumulator, bins=10)),
                       "Clients/selection": wandb.Histogram(np_histogram=np.histogram(clients_selected, bins=10)),
                       "Clients/success": wandb.Histogram(np_histogram=np.histogram(clients_success, bins=10)),
                       "Clients/fail": wandb.Histogram(np_histogram=np.histogram(clients_fail, bins=10)),
                       }, step=self.epoch)

    def get_client_conf(self, clientId):
        # learning rate scheduler
        conf = {}
        conf['learning_rate'] = self.args.learning_rate
        return conf

    def event_monitor(self):
        logging.info("Start monitoring events ...")

        while True:
            if len(self.event_queue) != 0:
                event_msg = self.event_queue.popleft()
                send_msg = {'event': event_msg}

                if event_msg == 'update_model':
                    self.broadcast_msg(send_msg)
                    self.broadcast_models()

                elif event_msg == 'start_round':
                    for executorId in self.executors:
                        next_clientId = self.resource_manager.get_next_task()
                        if next_clientId is not None:
                            config = self.get_client_conf(next_clientId)
                            self.server_event_queue[executorId].put({'event': 'train', 'clientId':next_clientId, 'conf': config})

                elif event_msg == 'stop':
                    self.broadcast_msg(send_msg)
                    self.stop()
                    break

                elif event_msg == 'report_executor_info':
                    self.broadcast_msg(send_msg)

                elif event_msg == 'test':
                    self.broadcast_msg(send_msg)

                #Ahmed - handle the case when round failures is disabled and no client meets the deadline
                elif event_msg == 'skip_round':
                    # Ahmed - log round metrics and reset
                    self.log_round_metrics()
                    self.round_reset_metrics()
                    self.round_completion_handler()

            elif not self.client_event_queue.empty():

                event_dict = self.client_event_queue.get()
                event_msg, executorId, results = event_dict['event'], event_dict['executorId'], event_dict['return']

                if event_msg != 'train_nowait':
                    logging.info(f"Round {self.epoch}: Receive (Event:{event_msg.upper()}) from (Executor:{executorId})")

                # collect training returns from the executor
                if event_msg == 'train_nowait':
                    # pop a new client to run
                    next_clientId = self.resource_manager.get_next_task()

                    if next_clientId is not None:
                        config = self.get_client_conf(next_clientId)
                        runtime_profile = {'event': 'train', 'clientId':next_clientId, 'conf': config}
                        self.server_event_queue[executorId].put(runtime_profile)

                elif event_msg == 'train':
                    # Ahmed - calculate the importance of the updates
                    #importance = 1.0/self.total_updates
                    total_num = ( self.args.stale_factor * self.tasks_round) + self.round_stale_updates
                    new_importance  =  self.args.stale_factor / total_num
                    stale_importance = 1.0 /total_num

                    # push training results
                    self.client_completion_handler(results, importance=new_importance)

                    # Ahmed - perform handler if we have enough number of clients
                    total_clients = self.tasks_round
                    if args.stale_update > 0:
                        total_clients += len(self.strugglers_to_run)
                    if len(self.stats_util_accumulator) == total_clients:
                        logging.info(f'====== Aggregator round complete - received {total_clients} new {self.tasks_round} struggler {len(self.round_stragglers)}')
                        # Ahmed - invoke the stale clients handler first
                        self.stale_clients_handler(importance=stale_importance)

                        # Ahmed - restore the model of last round if round failed
                        # if not round_update:
                        #     for idx, param in enumerate(self.model.parameters()):
                        #         param.data = self.last_global_model[idx]

                        self.round_completion_handler()

                        #Ahmed - log round metrics and reset
                        self.log_round_metrics()
                        self.round_reset_metrics()

                elif event_msg == 'test':
                    self.testing_completion_handler(results)

                elif event_msg == 'report_executor_info':
                    self.executor_info_handler(executorId, results)

                #Ahmed - stop the aggregator if one of the executors report trouble
                elif event_msg == 'executor_error':
                    logging.info("Aggregator received error message from executor".format(executorId))
                    self.broadcast_msg('emergency_stop')
                    logging.info("Aggregator broadcasted emergency stop due to error to executors")
                    time.sleep(5)
                    self.stop()
                    exit(0)
                else:
                    logging.error("Unknown message types!")

            # execute every 100 ms
            time.sleep(0.1)

    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)
        self.control_manager.shutdown()

if __name__ == "__main__":


    aggregator = Aggregator(args)

    ############ Initiate WANDB ###############
    if args.use_wandb:
        # Ahmed - set and init wandb run
        project = args.data_set + '_' + args.model.split('_')[0]
        gradient_policy = '_YoGi' if args.gradient_policy == 'yogi' else '_QFedAvg' if args.gradient_policy == 'qfedqvg' else '_Prox' if args.gradient_policy == 'prox' else '_FedAvg'
        run_name = 'Exp' + str(args.exp_type) + '_' + str(args.sample_mode) + str(gradient_policy) + '_S' + str(args.stale_update) + '_P' + str(args.avail_priority)\
                   + '_N' + str(args.total_worker) + '_D' + str(int(args.deadline)) + '_T' + str(args.target_ratio) + '_O' + str(args.overcommitment) +'_' + str(args.sample_seed) \
                   #+ '_R' + str(args.epochs) + '_E' + str(args.local_steps) + '_B' + str(args.batch_size)
        if args.resume_wandb:
            wandb_run = wandb.init(project='energy_eff', entity='flsys_qmul', name=run_name,
                                   config=dict(args.__dict__), id=run_name + '_' + args.time_stamp, resume=True)
        else:
            wandb_run = wandb.init(project='energy_eff', entity='flsys_qmul', name=run_name,
                                   id=run_name + '_' + args.time_stamp, config=dict(args.__dict__))

    ############################################
    aggregator.run()

