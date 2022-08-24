import os
import sys
import argparse

job_name = sys.argv[1]

if job_name == 'all':
    os.system("ps -ef | grep python | grep EAFL > eafl_running_temp")
else:
    os.system("ps -ef | grep python | grep job_name={} > eafl_running_temp".format(job_name))
[os.system("kill -9 "+str(l.split()[1]) + " 1>/dev/null 2>&1") for l in open("eafl_running_temp").readlines()]
os.system("rm eafl_running_temp")
