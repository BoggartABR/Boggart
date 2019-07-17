import os
import time
from subprocess import Popen
import numpy as np
import random

algorithms = ["Boggart", "Bola", "Pensieve", "MPC", "Throughput", "Buffer"]

# plays the video with all the algorithms at the same time, with 10 seconds interval for initializing.
# each video plays for 320 seconds
# before running, need to run init_all_servers
for i in range(50):

    perm = np.random.permutation(len(algorithms))
    processes = []
    for j in perm:
        processes.append(Popen("python2 test_on_internet/communication.py " + algorithms[j], shell=True))
        time.sleep(10)
    for p in processes:
        p.communicate()