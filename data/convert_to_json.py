import json
import os

DIR = 'fcc/'
OUT_DIR = 'json_fcc/'
LATENCY_IN_MS = 80.0

# converts trace format of traces in 'data' folder to json format used in this project and in Sabre.
for trace in os.listdir(DIR):
    trace_arr = []
    with open(DIR + trace) as t:
        lines = t.readlines()
    prev_time = 0.0
    for i in range(len(lines)):
        trace_arr.append({})
        in_time = float(lines[i].split()[0]) * 1000.0
        trace_arr[i]["duration_ms"] = in_time - prev_time
        prev_time = in_time
        trace_arr[i]["bandwidth_kbps"] = float(lines[i].split()[1]) * 1000.0 / 8.0
        trace_arr[i]["latency_ms"] = LATENCY_IN_MS
    with open(OUT_DIR + trace, 'ab') as out:
        json.dump(trace_arr, out)


