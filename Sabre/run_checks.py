import os

qoe = 'log'
traces = 'norway'
DIR = 'test_traces_json/'
VIDEO_FILE = 'manifest.json'
alg = 'pensieve'
for trace in os.listdir(DIR):
    os.system('python3 sabre.py -n ' + DIR+trace + ' -m ' + VIDEO_FILE + ' -b 60.0 -noa -a ' + alg + ' -tr ' + traces + ' -qoe ' + qoe)

