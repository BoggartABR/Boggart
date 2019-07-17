import numpy as np
import itertools
from constants import *
from abr_algs.abr import ABR

CHUNK_COMBO_OPTIONS = []
THPT_HISTORY = 5
MPC_FUTURE_CHUNK_COUNT = 5

# past errors in bandwidth
# make chunk combination options
for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
    CHUNK_COMBO_OPTIONS.append(combo)


class MPC(ABR):

    def __init__(self, video, reward):
        super(MPC, self).__init__(video, reward)
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.reward = reward

    def get_chunk_size(self, quality, index):
        if index < 0 or index >= self.video[NUM_OF_CHUNKS]:
            return 0
        return self.video[VIDEO_SIZES][index][quality]    # in mbyte

    def get_quality(self, network_state):

        # ================== MPC ========================= #
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0):
            curr_error = abs(self.past_bandwidth_ests[-1] - network_state[THROUGHPUT_IDX, -1]) / float(network_state[THROUGHPUT_IDX, -1])
        self.past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        self.past_bandwidths = network_state[THROUGHPUT_IDX, -THPT_HISTORY:]
        while self.past_bandwidths[0] == 0.0:
            self.past_bandwidths = self.past_bandwidths[1:]

        bandwidth_sum = 0
        for past_val in self.past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(self.past_bandwidths))


        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        error_pos = -5
        if (len(self.past_errors) < THPT_HISTORY):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        # future chunks length (try 4 if that many remaining)
        last_index = int(self.video[NUM_OF_CHUNKS] - network_state[CHUNKS_TILL_END_IDX, -1])
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (self.video[NUM_OF_CHUNKS] - last_index < MPC_FUTURE_CHUNK_COUNT):
            future_chunk_length = self.video[NUM_OF_CHUNKS] - last_index


        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = network_state[BUFFER_IDX, -1]
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int(network_state[LAST_BR_IDX, -1])
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (self.get_chunk_size(chunk_quality, index) * M_IN_K) / future_bandwidth

                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += self.video[VIDEO_CHUNK_LENGTH]
                bitrate_sum += self.reward.mapping(self.video[BITRATE_LEVELS][chunk_quality])
                smoothness_diffs += abs(self.reward.mapping(self.video[BITRATE_LEVELS][chunk_quality]) -
                                        self.reward.mapping(self.video[BITRATE_LEVELS][last_quality]))
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            reward = bitrate_sum - (self.reward.rebuf_penalty * curr_rebuffer_time) - smoothness_diffs

            if (reward >= max_reward):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if (best_combo != ()):  # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data

        return bit_rate

    def update(self, reward):
        return


