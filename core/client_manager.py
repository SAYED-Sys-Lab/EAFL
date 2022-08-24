from helper.client import Client
import math
from random import Random
import pickle
import logging, oort
from oort import *



class clientManager(object):
    def __init__(self, mode, args, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucbSampler = None 

        if self.mode == 'oort':
            from oort import create_training_selector
            self.ucbSampler = create_training_selector(args=args)

        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        #Ahmed - cache online clients
        self.cur_alive_clients = []
        # Ahmed - introduce a per client counter of avail periods equal to the length of deadline
        self.avail_counter = {}
        self.low_avail = 0

            #Ahmed - set values for the random assignment from low avail clients
        #if self.args.random_behv == 2:
         #   self.sorted_user_ids = [item[0] for item in sorted(self.user_trace.items(), key=lambda item: item[1]['duration'])]
          #  self.user_trace_len = len(self.user_trace)

    #Ahmed - return the availability of a client in a certain time window (time slots = deadline)
    def isAvailable(self, clientId, cur_time, time_window, deadline):
        start_time=cur_time + (time_window-1) * deadline
        end_time=cur_time + time_window * deadline

        availabilityPeriods=self.Clients[self.getUniqueId(0, clientId)].availabilityPeriods()
        #logging.info('==== Client {} - start_time {}, end_time {} avails {}'.format(clientId, start_time,end_time, availabilityPeriods))
        for period in availabilityPeriods:
            start, end = period
            if start <= start_time and end >= end_time:
                return True
        return False

    #Ahmed - return the priority of a client based on its availability in a certain time window (time slots = deadline)
    def getPriority(self,clientId, cur_time, deadline, time_window=3):
        priority=0
        for i in range(time_window, 0, -1):
            if self.isAvailable(clientId, cur_time, deadline, i):
                    priority=time_window - i
                    break
        return priority

    # Ahmed - get the count of availability periods divided into slots of the deadline
    def getPeriodCount(self, clientId, cur_time, deadline):
        availabilityPeriods = self.Clients[self.getUniqueId(0, clientId)].availabilityPeriods()
        finishtime = self.Clients[self.getUniqueId(0, clientId)].traces['finish_time']
        norm_time = cur_time % finishtime
        index = 0
        for period in availabilityPeriods:
            start, end = period
            if norm_time < start:
                # logging.info('client {} period {} normtime {}'.format(clientId, period, norm_time))
                break
            index += 1
        count = 0
        if index > 0:
            v1, v2 = availabilityPeriods[index - 1]
            count += int((v2 - v1 - norm_time) / deadline)
        for i in range(index, len(availabilityPeriods)):
            start, end = availabilityPeriods[i]
            duration_normed = int((end - start) / deadline)
            if duration_normed > 0:
                count += duration_normed
        return count


    def registerClient(self, hostId, clientId, size, speed, category, duration=1):
        uniqueId = self.getUniqueId(hostId, clientId)
        #user_trace = None if self.user_trace is None else self.user_trace[int(clientId)]
        if self.user_trace:
            if self.args.random_behv > 0:
                if self.args.random_behv == 1:
                    # randomly set the user behaviour to the client
                    user_trace = self.user_trace[self.rng.randint(1, len(self.user_trace))]
                elif self.args.random_behv == 2:
                    u = self.rng.random()
                    index = int(self.user_trace_len * 0.1)
                    if u < 1.0:
                        user_id = self.rng.choice(self.sorted_user_ids[:index])
                        self.low_avail += 1
                    else:
                        user_id = self.rng.choice(self.sorted_user_ids[index + 1:])
                    user_trace = self.user_trace[user_id]
            else:
                # Ahmed - fix an error thrown when the clientId from the dataset is more than 107K in user_trace (happend with stackoverflow)
                # Ahmed - for stackoverflow feasible clients {'total_feasible_clients': 281347, 'total_length': 41367564}
                # set the user behaviour based on client ID (sequential)
                cid = int(clientId) % len(self.user_trace)
                if cid in self.user_trace:
                    user_trace = self.user_trace[cid]
                else:
                    cid=self.rng.randint(1, len(self.user_trace))
                    user_trace = self.user_trace[cid]

        else:
            user_trace = None

        self.Clients[uniqueId] = Client(hostId, clientId, speed, category, user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(clientId)
            self.feasible_samples += size

            if self.mode == 'oort':
                time = self.getCompletionTime(clientId, self.args.batch_size, self.args.local_steps, 0, 0)
                reward = min(size, self.args.local_steps * self.args.batch_size)
                feedbacks = {'reward':reward,
                            'duration': duration,
                            }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)

    def setbatterylevel(self, clientId, category):

            self.Clients[self.getUniqueId(0, clientId)].setbatterylevel(category=category)

    def getbattery(self, clientId):

            return self.Clients[self.getUniqueId(0, clientId)].getbattery()

    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)]

    def registerDuration(self, clientId, batch_size, upload_epoch, upload_size, download_size):
        if self.mode == 'oort':
            exe_cost = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                    batch_size=batch_size, upload_epoch=upload_epoch,
                    upload_size=upload_size, download_size=download_size
            )
            self.ucbSampler.update_duration(clientId, exe_cost['computation']+exe_cost['communication'])


    def getEnergy(self, clientId, comp, comm):
            energy = self.Clients[self.getUniqueId(0, clientId)].getEnergy(comp, comm)
            return energy

    def getenergyeff(self, clientId, energy, batch_size):
            energyeff = self.Clients[self.getUniqueId(0, clientId)].getEnergyeff(energy, batch_size=batch_size)
            return energyeff

    def updateutility(self, clientId, energy):
            update = self.Clients[self.getUniqueId(0, clientId)].updateutility(energy)
            return update

    def getCompletionTime(self, clientId, batch_size, upload_epoch, upload_size, download_size):
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_epoch=upload_epoch,
                upload_size=upload_size, download_size=download_size
            )

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward

        if self.mode == 'oort':
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }
            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)
        self.registerClientScore(clientId, reward)



    def registerClientScore(self, clientId, reward):
        self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId - 1
        lenPossible = len(self.feasibleClients)

        while True:
            clientId = str(self.feasibleClients[init_id])
            csize = self.Clients[clientId].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(clientId)
            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))
        return init_id

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds

    def getCurrentClientIds(self, hostId):
        return self.clientOnHosts[hostId]

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])

    def getClientSize(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.
        if not even:
            for key in self.clientOnHosts.keys():
                for client in self.clientOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.Clients[uniqueId].size

            #1./len(self.clientOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)
        else:
            for key in self.clientOnHosts.keys():
                totalSampleInTraining += len(self.clientOnHosts[key])

            return 1./totalSampleInTraining

    def getFeasibleClients(self, cur_time):

        clients_alive = [clientId for clientId in self.feasibleClients if self.Clients[self.getUniqueId(0, clientId)].getbatterystatus()]

        return clients_alive
        #logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_alive)} clients alive, " + \
                    #f"{len(self.feasibleClients)-len(clients_alive)} clients dead")


    def isClientAlive(self, clientId, roundDuration):
        return self.Clients[self.getUniqueId(0, clientId)].isAlive(roundDuration)

    #Ahmed - introduce function for returning online and offline clients
    # def getOnlineClients(self, cur_time):
    #     if self.user_trace is None:
    #         return self.feasibleClients, None
    #     online_clients = []
    #     offline_clients = {}
    #     for clientId in self.feasibleClients:
    #         active, bindex = self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)
    #         if active:
    #             online_clients.append(clientId)
    #         else:
    #             offline_clients[clientId] = bindex
    #     return online_clients, offline_clients

    def resampleClients(self, numOfClients, cur_time=0, time_window=0):

        self.count += 1

        clients_alive = self.getFeasibleClients(cur_time)
        self.cur_alive_clients = clients_alive

        if len(clients_alive) <= numOfClients:
            return clients_alive

        feasible_clients = clients_alive
        #########################################
        # Ahmed - add clients to the counter list
        #count_ok = False
        #if self.args.avail_priority > 1:
        #    count_ok = True
        #    for c in feasible_clients:
         #       if c not in self.avail_counter:
          #          if time_window > 0:
           #             self.avail_counter[c] = self.getPeriodCount(c, cur_time, time_window)
            #        else:
             #           count_ok = False
              #          break
               # elif self.avail_counter[c] > 0:
                #    self.avail_counter[c] -= 1

        #online = len(feasible_clients)
        #target_num = numOfClients  # max(numOfClients, int(0.25 * online))

        # Ahmed - select from the high priority with p=2
        # TODO: create a sub-set of feasible (online) with the ones that have priority = 2
        #if self.args.avail_priority == 1:
         #   priority_vals = {}
          #  for c in feasible_clients:
           #     priority_vals[c] = self.getPriority(c, cur_time, time_window, time_window=2)
           # priority_clients = [key for key, val in priority_vals.items() if val == 1]
            #remaining_clients = [key for key, val in priority_vals.items() if val == 0]
            #feasible_clients = [v for v in priority_clients]
            #while len(feasible_clients) < target_num:
             #   feasible_clients.append(self.rng.choice(remaining_clients))
            #logging.info(
             #   "==== PS Client Sampler - avail prio 1 - online {}, feasible {}, high priority {}:{}, remaining {}:{} ".format(
              #      online, len(feasible_clients), \
               #     len(priority_clients),
                #    {c:self.Clients[self.getUniqueId(0, c)].traces['duration'] for c in priority_clients}, \

                 #   len(remaining_clients),
                  #  {c:self.Clients[self.getUniqueId(0, c)].traces['duration'] for c in remaining_clients}))
        #elif count_ok and self.args.avail_priority == 2:
         #   clients_to_select = {k: v for k, v in self.avail_counter if k in feasible_clients}
          #  vals = np.array(list(clients_to_select.values()), dtype=np.float64)
           # valsmean = vals.mean()
            #probs = 1 - (vals / valsmean)
           # probs_new = np.where(probs < 0, 0, probs)
           # probs_norm = probs_new / probs_new.sum()
           # # probs = {key:(1.0 - (1.0 * val / count_sum)) for key, val in count_vals.items()}
            #if len(probs) > 0:
             #   feasible_clients = np.random.choice(list(clients_to_select.keys()), size=numOfClients, p=probs_norm,
             #                                       replace=False)
           # # logging.info("==== PS Client Sampler - avail prio 2 - online {}, feasible {} Sum {} probsum {} vals {} Prob {} Probnorm{} ".format(online, len(feasible_clients), valsmean,  probs.sum(), vals.tolist(), probs.tolist(), probs_norm.tolist()))
            #logging.info("==== PS Client Sampler - avail prio 2 - online {}, feasible {} ProbNorm {} ".format(online,
             #                                                                                                 len(
              #                                                                                                    feasible_clients),
               #                                                                                               probs_norm.tolist()))

            ## sorted_count_vals = dict(sorted(count_vals.items(), key=lambda item: item[1]))
            ## sorted_clients = sorted(count_vals, key=count_vals.get)
            ## feasible_clients = sorted_clients[:numOfClients]
            ## min_c = feasible_clients[0]
            ## max_c = feasible_clients[-1]
            ## logging.info("==== PS Client Sampler - avail prio 2 - online {}, feasible {}, Min {}:{}, Max {}:{}, Count {}".format(online, len(feasible_clients),\
            #                                                                                                 min_c, count_vals[min_c], max_c, count_vals[max_c], sorted_count_vals))
        #elif count_ok and self.args.avail_priority == 3:
         #   clients_to_select = {}
          #  avg = 0
           # count = 0
            #for k, v in self.avail_counter.items():
             #   if k in feasible_clients:
              #      clients_to_select[k] = v
               #     avg += v
                #    count += 1
            #mean_val = 1.0 * avg / count
            ## mean_val = np.mean(list(clients_to_select.values()))

            #feasible_clients = []
            #for k, v in clients_to_select.items():
             #   if v < mean_val:
              #      feasible_clients.append(k)
            ## vals = np.array(list(clients_to_select), dtype=np.float64)
            ## probs = 1 - (vals / vals.mean())
            ## mask = np.where(probs <= 0, 0, probs)
            ## mask = np.where(mask > 0, 1, mask)
            ## feasible_clients = np.ma.masked_array(np.array(feasible_clients), mask)
            ## feasible_clients = feasible_clients.compressed().tolist()
            #logging.info(
             #   "==== PS Client Sampler - avail prio 3 - online {}, feasible {}".format(online, len(feasible_clients)))

        ## if len(feasible_clients) <= target_num:
        ##     return feasible_clients
        ##########################################

        pickled_clients = None
        feasible_clients_set = set(feasible_clients)
        if self.mode == 'oort' and self.count > 1:
            pickled_clients = self.ucbSampler.select_participant(numOfClients, feasible_clients=feasible_clients_set)
        else:
            self.rng.shuffle(feasible_clients)
            client_len = min(numOfClients, len(feasible_clients) -1)
            pickled_clients = feasible_clients[:client_len]
        return pickled_clients

    def getAllMetrics(self):
        if self.mode == 'oort':
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, clientId):
        return self.ucbSampler.get_client_reward(clientId)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucbSampler.get_median_reward()
        return 0.

    def setbatterystatus(self, clientId, status):
        self.Clients[self.getUniqueId(0, clientId)].setbatterystatus(status=status)

    def idlepowerdeduction(self, clients, duration):
        for clientId in self.feasibleClients:
            if clientId not in clients and self.Clients[self.getUniqueId(0, clientId)].getbatterystatus():
                battery=self.Clients[self.getUniqueId(0, clientId )].idlepowerdeduction(duration=duration)










