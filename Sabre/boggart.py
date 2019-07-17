import numpy as np
import contextual_exp3
import pickle
import json
import random

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
CHUNK_TIL_VIDEO_END_CAP = 48.0 #116.0
HISTORY_LEN = 8
INPUT_LEN = 11

HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
A_DIM = len(VIDEO_BIT_RATE)

M_IN_K = 1000.0
REBUF_PENALTY = max(VIDEO_BIT_RATE) / M_IN_K  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
MAX_REBUF = 15
BUFFER_THRESH = 60.0

RESEVOIR = 5  # BB
CUSHION = 10  # BB
LOAD_FILE = None
THPT_HIST_LENGTH = 3
CONTEXT_DIM = (7,7)
PREDICTIONS = np.array([-(A_DIM - 1), -np.floor(np.sqrt(A_DIM - 1)), -1, 0, 1, np.floor(np.sqrt(A_DIM - 1)), (A_DIM - 1)]).astype(int)
WEIGHTS = 'exp3_models/inc_dec_log_weights.json'
GAMMAS = 'exp3_models/inc_dec_log_gamma.json'

random.seed(0)

def get_context(last_quality, buffer_size, throughput, next_sizes):


    next_sizes = np.array(next_sizes)
    qualities_idx = np.array(np.minimum(np.maximum(last_quality + PREDICTIONS, 0), A_DIM-1))
    qualities = next_sizes[qualities_idx] / 4.0
    tmp = (np.zeros(len(qualities)) + throughput) - qualities
    tmp = tmp[tmp > 0]
    throughput_idx = max(len(tmp) - 1, 0)

    download_time = next_sizes / throughput     # * 1000.0)
    download_time = np.array(download_time)[qualities_idx]
    tmp = (np.zeros(len(qualities_idx)) + buffer_size) - download_time
    tmp = tmp[tmp > 0]
    buffer_idx = max(len(tmp) - 1, 0)

    return throughput_idx, buffer_idx


class Boggart:

    def __init__(self, qoe):
        self.gammas = 'exp3_models/inc_dec_' + qoe + '_gamma.json'
        self.weights = 'exp3_models/inc_dec_' + qoe + '_weights.json'
        with open(self.weights) as wf:
            weights = json.load(wf)
        with open(self.gammas) as gf:
            gammas = json.load(gf)
        self.trainer = contextual_exp3.Contextual_Exp3((7,7), 7)
        self.trainer.load_weights(weights, gammas)

    def predict(self, last_quality, throughput, buffer, next_sizes):
        context = get_context(last_quality, buffer, throughput, next_sizes)
        prediction = self.trainer.predict(context)
        quality = max(min(last_quality + PREDICTIONS[prediction], len(next_sizes) - 1), 0)
        return quality