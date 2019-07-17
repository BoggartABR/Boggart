import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

LINK_FILE = 'fcc_trace0'
DIR = 'fcc/'


with open(DIR + LINK_FILE) as f:
	lines = f.readlines()


times = []
bw = []

harmonic = []
exp = []
alpha = 0.5
for l in lines:

	if len(bw) == 0:
		exp.append(0)
		harmonic.append(0)
	else:
		if len(exp) == 0:
			exp.append(bw[-1])
		else:
			exp.append(alpha * bw[-1] + (1. - alpha) * exp[-1])
		harmonic.append(float(min(8, len(bw))) / sum(1. / np.array(bw[-min(8, len(bw)) : ])))

	split = l.split()
	times.append(float(split[0]))
	thpt = float(split[1])
	bw.append(thpt)


plt.plot(bw[20:60], '-r', exp[20:60], 'g:', harmonic[20:60], 'b--')
plt.xlabel('Time (second)')
plt.ylabel('Throughput (Mbit/sec)')
plt.legend(["Bandwidth", "Exponential discount","Harmonic mean"])
plt.show()
