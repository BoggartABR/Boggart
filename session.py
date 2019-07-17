import numpy as np
import os
import sys

from env import Environment
import rewards
from constants import *
import json
from abr_algs import bolae, bba, mpc, throughput
from abr_algs.boggart import boggart
from abr_algs.pensieve import Pensieve

##
##  Runs all the traces from trace_dir with abr_type algorithm
##

VIDEO = 'manifest.json'
BOGGART_MODEL = 'abr_algs/boggart/json_models/inc_dec'
NN_MODEL = 'abr_algs/pensieve/models/lin_model.ckpt'

# time interval to save parameters of trained boggart model
MODEL_SAVE_INTERVAL = 100000

# Options for ABR algorithm:
#   - BOGGART
#   - PENSIEVE
#   - MPC
#   - BBA
#   - THROUGHPUT
#   - BOLA

abr_type = sys.argv[1]

# directory with test / train traces
trace_dir = sys.argv[2]

# Options for reward type: LIN, LOG, HD
reward_type = sys.argv[3]

# logging directory
LOG_DIR = sys.argv[4]


# Options for environment type: TRAIN (only for boggart), TEST
# TRAIN type chooses random traces and never terminates, TEST type plays the traces in order and haults.
env_type = TEST

log_file = LOG_DIR + abr_type + "_" + trace_dir + "_" + reward_type


# initializes parameters of played video
def init_video(video_file):

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


# loads the traces from traces directory and converts them to readble format
def load_trace(trace_folder):

    traces = {}

    cooked_files = os.listdir(trace_folder)
    cooked_files.sort()
    traces[TIME] = []
    traces[BANDWIDTH] = []
    traces[FILE_NAMES] = []
    for cooked_file in cooked_files:
        file_path = trace_folder + '/' + cooked_file
        cooked_time = [0]
        cooked_bw = []
        with open(file_path) as f:
            trace_contents = json.load(f)
            for t in trace_contents:
                cooked_time.append(cooked_time[-1] + (t["duration_ms"] / M_IN_K))   # seconds
                cooked_bw.append(t["bandwidth_kbps"] / M_IN_K ) # in mbits
        cooked_time = cooked_time[1:]
        assert(len(cooked_time) == len(cooked_bw))
        traces[TIME].append(cooked_time)
        traces[BANDWIDTH].append(cooked_bw)
        traces[FILE_NAMES].append(cooked_file)

    return traces


# factory of ABR algorithms
def load_ABR(type, video, save_file, reward):

    if type == BOGGART:
        return boggart.Boggart(video, reward, BOGGART_MODEL)
    if type == PENSIEVE:
        return Pensieve.Pensieve(video, reward, NN_MODEL)
    if type == BOLA:
        return bolae.BolaEnh(video, reward)
    if type == BBA:
        return bba.BBA(video, reward)
    if type == THROUGHPUT:
        return throughput.Throughput(video, reward)
    if type == MPC:
        return mpc.MPC(video, reward)


video = init_video(VIDEO)
reward = rewards.get_reward_class(video, reward_type)
alg = load_ABR(abr_type, video, log_file, reward)
traces = load_trace(trace_dir)
env = Environment(traces, env_type, video, RANDOM_SEED)

quality = DEFAULT_QUALITY
last_quality = DEFAULT_QUALITY
state = np.zeros((INPUT_LEN, HISTORY_LEN))
video_count = 0
t = 0

# for logging the results
rewards = []
qualities = []
rebufs = []
jitter = []
while True:

    last_batch, sleep_time, rebuf, end_of_video = env.get_video_chunk(quality)
    remaining_chunks = last_batch[CHUNKS_TILL_END_IDX]

    # Boggart trains with normalized reward
    norm_reward = reward.norm_reward(quality, last_quality, rebuf)
    immediate_reward = reward.reward(quality, last_quality, rebuf)

    if env_type == TRAIN and abr_type == BOGGART:
        alg.update(norm_reward)

    first_round = False
    their_reward = reward.reward(quality, last_quality, rebuf)

    # add to log lists
    rewards.append(immediate_reward)
    qualities.append(reward.mapping(video[BITRATE_LEVELS][quality]))
    rebufs.append(rebuf)
    jitter.append(np.abs(reward.mapping(video[BITRATE_LEVELS][quality]) -
                                        reward.mapping(video[BITRATE_LEVELS][last_quality])))
    last_quality = quality

    # convert to format that Pensieve receives as input
    state = np.roll(state, -1, axis=1)
    state[LAST_BR_IDX, -1] = last_batch[LAST_BR_IDX]
    state[BUFFER_IDX, -1] = last_batch[BUFFER_IDX]
    state[THROUGHPUT_IDX, -1] = last_batch[THROUGHPUT_IDX]
    state[DELAY_IDX, -1] = last_batch[DELAY_IDX]
    state[NEXT_CHUNKS_START_IDX, :video[BR_DIM]] = \
        last_batch[NEXT_CHUNKS_START_IDX:NEXT_CHUNKS_START_IDX + video[BR_DIM]]
    state[CHUNKS_TILL_END_IDX, -1] = last_batch[NEXT_CHUNKS_START_IDX + video[BR_DIM]]

    quality = alg.get_quality(state)

    if end_of_video:
        with open(log_file, 'ab') as log:
            log_dict = {}
            log_dict["qualities"] = np.average(qualities)
            log_dict["rebufs"] = np.average(rebufs)
            log_dict["jitter"] = np.average(jitter)
            log_dict["rewards"] = np.average(rewards)
            json.dump(log_dict, log)
            log.write('\n')

        last_quality = DEFAULT_QUALITY
        quality = DEFAULT_QUALITY
        video_count += 1
        rewards = []
        qualities = []
        rebufs = []
        jitter = []

    if env_type == TEST and video_count > len(traces[FILE_NAMES]):
        break
    t += 1

    # update trained model parameters
    if t % MODEL_SAVE_INTERVAL == 0 and env_type == TRAIN and abr_type == BOGGART:
        alg.save_params()
