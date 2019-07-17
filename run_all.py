import os
from constants import *
import json
import numpy as np

abr_algs = [BOGGART, PENSIEVE, THROUGHPUT, MPC, BOLA, BBA]
qoe = LIN
DIR = 'one_trace'
LOG_DIR = 'results/'


# run all the algorithms on traces from DIR and display results
for alg in abr_algs:
    if os.path.isfile(LOG_DIR + alg + "_" + DIR + "_" + qoe):
        os.remove(LOG_DIR + alg + "_" + DIR + "_" + qoe)
    os.system('python2 session.py ' + alg + ' ' + DIR + ' ' + qoe + ' ' + LOG_DIR)

for file in os.listdir(LOG_DIR):
    rewards = []
    jitter = []
    qualities = []
    rebufs = []
    with open(LOG_DIR + file) as f:
        lines = f.readlines()
    for l in lines:
        result_dict = json.loads(l)
        rewards.append(result_dict["rewards"])
        jitter.append(result_dict["jitter"])
        rebufs.append(result_dict["rebufs"])
        qualities.append(result_dict["qualities"])
    print "rewards for " + file + ":", np.average(rewards)
    print "jitter for " + file + ":", np.average(jitter)
    print "qualities for " + file + ":", np.average(qualities)
    print "rebufs for " + file + ":", np.average(rebufs)
    print
