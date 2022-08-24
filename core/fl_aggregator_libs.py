# package for aggregator
from fllibs import *

#logDir = os.path.join(os.environ['HOME'], "models", args.model, args.time_stamp, 'aggregator')
time_stamp = args.time_stamp
logDir = os.path.join(args.log_path, "models", args.model, time_stamp, 'aggregator')
if os.path.isdir(logDir):
    if 'SLURM_JOBID' in os.environ:
        time_stamp += '/' + os.environ['SLURM_JOBID']
    else:
        time_stamp += "_" + str(random.randint(1,100))
    logDir = os.path.join(args.log_path, "models", args.model, time_stamp, 'aggregator')
logFile = os.path.join(logDir, 'log')

def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    logging.basicConfig(
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='(%m-%d) %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

def dump_ps_ip():
    hostname_map = {}
    with open('ipmapping', 'rb') as fin:
        hostname_map = pickle.load(fin)

    ps_ip = str(hostname_map[str(socket.gethostname())])
    args.ps_ip = ps_ip

    with open(os.path.join(logDir, 'ip'), 'wb') as fout:
        pickle.dump(ps_ip, fout)

    logging.info(f"Load aggregator ip: {ps_ip}")


def initiate_aggregator_setting():
    init_logging()
    #dump_ps_ip()

initiate_aggregator_setting()
