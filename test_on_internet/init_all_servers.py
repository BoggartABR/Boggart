from constants import *
import rewards
from servers import mpc_server, pensieve_server, logging_server, server
import os
import json
import numpy as np
from multiprocessing import Process

VIDEO_FILE = 'manifest.json'
reward_type = LIN


# initializes parameters of played video
def init_video(video_file = VIDEO_FILE):

    with open(video_file) as vid:
        manifest = json.load(vid)

    video = {}
    video[NUM_OF_CHUNKS] = len(manifest["segment_sizes_bits"])
    video[BITRATE_LEVELS] = manifest["bitrates_kbps"]
    video[HD_REWARD] = manifest["hd_rewards"]
    video[VIDEO_CHUNK_LENGTH] = manifest["segment_duration_ms"] / M_IN_K    # in secs

    video[BR_DIM] = len(video[BITRATE_LEVELS])
    video[VIDEO_SIZES] = np.array(manifest["segment_sizes_bits"]) / M_IN_K / M_IN_K / BITS_IN_BYTE    # mega-byte
    return video


video = init_video(VIDEO_FILE)
reward = rewards.get_reward_class(video, reward_type)

processes = []
processes.append(Process(target=pensieve_server.run_server, args=[video, reward]))
processes.append(Process(target=mpc_server.run_server, args=[video, reward]))
processes.append(Process(target=logging_server.run_server))
processes.append(Process(target=server.run_server))

for p in processes:
    p.start()
for p in processes:
    p.join()

