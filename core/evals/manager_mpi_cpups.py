# Submit job to the remote cluster
#import yaml
import sys
import time
import random
import os, subprocess
import pickle, datetime
#Ahmed - import modules
import logging
#import wandb
from envyaml import EnvYAML
#from mpi4py import MPI


#mask = 0o771
#os.umask(0) #create the files with group permissions
#sys.setrecursionlimit(10000)

# node_list = os.environ['SLURM_JOB_NODELIST']
# gpus = os.environ['SLURM_JOB_GPUS']
# gpus_per_node = os.environ['SLURM_GPUS_PER_TASK']  # os.environ['SLURM_GPUS_PER_NODE']
# procs = os.environ['SLURM_NPROCS']
# rank = os.environ['SLURM_PROCID']
# run_dir = os.environ['SLURM_SUBMIT_DIR']
# run_node = os.environ['SLURMD_NODENAME']

# os.environ['HOME'] = '/ibex/scratch/sayeam0a'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# cuda_ids = os.environ['CUDA_VISIBLE_DEVICES']
# gpu_ids = os.environ['GPU_DEVICE_ORDINAL']

slurm_job_id = os.environ['SLURM_JOBID']


# print(node_list, gpus, gpus_per_node, procs, rank, run_dir, run_node)
#
# nodes = node_list.replace('[','').replace(']','').split(',')
# for i, node in enumerate(nodes):
#     print(node)
#     if node.find('gpu') < 0:
#         temp = nodes[i-1].split('-')[0]
#         nodename = temp + '-' + node
#         nodes.remove(node)
#         nodes.insert(i, nodename)
# print(nodes)
#
# ps_node = run_node
# ngpus_per_node = int(gpus_per_node) #int(procs)
# worker_nodes = []
# for i, node in enumerate(nodes):
#     n_name = node
#     n_gpu = ngpus_per_node
#     if i==0:
#         n_gpu = ngpus_per_node - 1
#     alloc = ''
#     for i in range(n_gpu):
#         if i == n_gpu - 1:
#             alloc += '1'
#         else:
#             alloc += '1,'
#     worker_node = n_name + ':[' + alloc + ']'
#     worker_nodes.append(worker_node)
# print(worker_nodes)

def load_yaml_conf(yaml_file):
    # with open(yaml_file) as fin:
    #     data = yaml.load(fin, Loader=yaml.FullLoader)
    data = EnvYAML(yaml_file)
    return data


def process_cmd(yaml_file, total_processes):
    yaml_conf = load_yaml_conf(yaml_file)

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        global_rank =0
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        local_rank=0

    ps_ip = os.environ['MASTER_ADDR']
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    job_name = ''
    log_path = './logs'
    job_conf = {'time_stamp': time_stamp,
                'total_worker': total_processes,
                'ps_ip': ps_ip,
                'ps_port': random.randint(10000, 60000),
                'manager_port': random.randint(10000, 60000)
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    if global_rank == 0:
        conf_script = ''
        setup_cmd = ''
        if yaml_conf['setup_commands'] is not None:
            setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
            for item in yaml_conf['setup_commands'][1:]:
                setup_cmd += (item + ' && ')

        cmd_sufix = f" "

        for conf_name in job_conf:
            conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
            if conf_name == "job_name":
                job_name = job_conf[conf_name]
            if conf_name == "log_path":
                log_main_path = job_conf[conf_name]
        print(f'======== PS log main path: {log_main_path}')
        log_path=os.path.join(log_main_path, 'logs', job_name, time_stamp)
        print(f'======== PS log path: {log_path}')
        #job_conf["log_path"] = log_path

        learner_conf = '-'.join([str(_) for _ in list(range(1 , total_processes + 1))])

        # dump the job config for the workers
        current_path = os.path.dirname(os.path.abspath(__file__))
        print(f'======== PS current path: {current_path}')
        log_path = os.path.join(current_path, 'logs', job_name, time_stamp)
        print(f'======== PS log path: {log_path}')

        job_conf_file = os.path.join(current_path, 'job_config', str(slurm_job_id))
        with open(job_conf_file, 'wb') as fout:
            pickle.dump(job_conf, fout)
        print(f'======== PS job config file written to {job_conf_file}')

        if not os.path.isdir(log_path):
            os.makedirs(log_path, exist_ok=True)
        else:
            os.makedirs(os.path.join(log_path, str(slurm_job_id)), exist_ok=True)
            log_path = os.path.join(log_path, str(slurm_job_id))
            print(f'======== PS log path switch to: {log_path}')

        job_conf_file = os.path.join(log_path, 'job_config')
        with open(job_conf_file, 'wb') as fout:
            pickle.dump(job_conf, fout)
        print(f'======== PS job config file written to {job_conf_file}')

        # =========== Submit job to parameter server ============
        with open(f"{log_path}/all_logs", 'wb') as fout:
            pass

        # Ahmed - account for the case when running on multi-node setting (e.g., ibex or EC2 for mpirun)
        if 'GPU_DEVICE_ORDINAL' in os.environ:
            gpu_ordinal = str(os.environ['GPU_DEVICE_ORDINAL']).split(',')
        else:
            gpu_ordinal = [0]
        gpu_max_ordinal = int(gpu_ordinal[-1])
        if gpu_max_ordinal < local_rank:
            local_rank = gpu_max_ordinal

        ps_cmd = f" python {yaml_conf['exp_path']}/aggregator.py {conf_script} --this_rank={global_rank} --learner={learner_conf} --cuda_device=cuda:{local_rank} --use_cuda=0"

        print(f"Starting aggregator {local_rank}:{global_rank} on {ps_ip}...")
        with open(f"{log_path}/all_logs", 'a') as fout:
            print(f'{setup_cmd} {ps_cmd}')
            #os.system(f'{setup_cmd} {ps_cmd}')
            subprocess.Popen(f'{setup_cmd} {ps_cmd}', shell=True,  stdout=fout, stderr=subprocess.STDOUT)

    #Ahmed - put them in else condition to exclude the PS GPU from the worker allocations

    #Worker processes wait till the PS starts and writes the job config
    # if job_conf['task'] == 'nlp':
    #     time.sleep(40)
    # else:
    time.sleep(20)

    #get the job name from the YAML file
    # configs = yaml_conf['job_conf']
    # for val in configs:
    #     if 'job_name' in val:
    #         job_name = val['job_name']
    #     if 'log_path' in val:
    #         log_main_path = val['log_path']
    #     if 'time_stamp' in val:
    #         time_stamp = val['time_stamp']

    # =========== Submit job to each worker ============
    # read the configs written by the PS server
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_conf_file = os.path.join(current_path, 'job_config', str(slurm_job_id))
    with open(job_conf_file, 'rb') as fin:
        job_conf = pickle.load(fin)

    job_name = ''
    log_main_path = './logs'
    time_stamp = ''
    #print(job_conf)
    for val in job_conf:
        if 'job_name' in val:
            job_name = job_conf[val]
        if 'log_path' in val:
            log_main_path = job_conf[val]
        if 'time_stamp' in val:
            time_stamp = job_conf[val]
    log_path = os.path.join(log_main_path, 'logs', job_name, time_stamp)
    if os.path.isdir(os.path.join(log_path, str(slurm_job_id))):
        log_path = os.path.join(log_path, str(slurm_job_id))

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    learner_conf = '-'.join([str(_) for _ in list(range(1, total_processes+1))])
    # =========== Submit job to each worker ============
    print(f"Starting worker {local_rank}:{global_rank+1} with PS: {job_conf['ps_ip']}...")

    # Ahmed - account for the case when running on multi-node setting (e.g., ibex or EC2 for mpirun)
    if 'GPU_DEVICE_ORDINAL' in os.environ:
        gpu_ordinal = str(os.environ['GPU_DEVICE_ORDINAL']).split(',')
    else:
        gpu_ordinal = [0]
    gpu_max_ordinal = int(gpu_ordinal[-1])
    if gpu_max_ordinal < local_rank:
        local_rank = gpu_max_ordinal

    worker_cmd = f" python {yaml_conf['exp_path']}/executor.py {conf_script} --this_rank={global_rank+1} --learner={learner_conf} --cuda_device=cuda:{local_rank}"
    with open(f"{log_path}/all_logs", 'a') as fout:
        print(f'{setup_cmd} {worker_cmd}')
        #os.system(f'{setup_cmd} {worker_cmd}')
        subprocess.call(f'{setup_cmd} {worker_cmd}', shell=True,  stdout=fout, stderr=subprocess.STDOUT)


def terminate(job_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        os.system(f"ssh {job_meta['user']}{vm_ip} 'python {current_path}/shutdown.py {job_name}'")


if sys.argv[1] == 'submit':
    process_cmd(sys.argv[2], int(sys.argv[3]))
elif sys.argv[1] == 'stop':
    terminate(sys.argv[2])
else:
    print("Unknown cmds ...")
