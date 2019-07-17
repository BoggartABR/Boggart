import numpy as np
import itertools

S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
# make chunk combination options
for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
    CHUNK_COMBO_OPTIONS.append(combo)


def get_qoe_mapping(type, quality):
    if type == 'lin':
        return VIDEO_BIT_RATE[quality] / 1000.0
    if type == 'log':
        return np.log(VIDEO_BIT_RATE[quality] / VIDEO_BIT_RATE[0])
    if type == 'hd':
        return HD_REWARD[quality]

class MPC:

    def __init__(self, param_function, qoe_type):

        self.size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522,
                       2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469,
                       2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074,
                       2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102,
                       2124950, 2246506, 1961140, 2155012, 1433658]
        self.size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548,
                       1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126,
                       1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081,
                       1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250,
                       1464921, 1483527, 1231456, 1364537, 889412]
        self.size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851,
                       1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935,
                       1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587,
                       908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
        self.size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282,
                       687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335,
                       696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884,
                       587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
        self.size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351,
                       434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700,
                       425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327,
                       390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
        self.size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746,
                       179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938,
                       181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254,
                       149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

        self.past_errors = []
        self.past_bandwidth_ests = []
        self.param_function = param_function
        self.qoe_type = qoe_type
        self.state = np.zeros((S_INFO, S_LEN))


    def get_chunk_size(self, quality, index):
        if (index < 0 or index > 48):
            return 0
        # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
        sizes = {5: self.size_video1[index], 4: self.size_video2[index], 3: self.size_video3[index], 2: self.size_video4[index],
                 1: self.size_video5[index], 0: self.size_video6[index]}
        return sizes[quality]


    def predict(self, last_quality, throughput, buffer_size, delay, next_sizes, video_chunk_remain, bitrates):

        self.state = np.roll(self.state, -1, axis=1)

        self.state[0, -1] = bitrates[last_quality] / float(np.max(bitrates))  # last quality
        self.state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.state[2, -1] = throughput  # k-byte to ms
        self.state[3, -1] = float(delay) / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, :A_DIM] = np.array(next_sizes)  # mega byte
        self.state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)


        # ================== MPC ========================= #
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0):
            curr_error = abs(self.past_bandwidth_ests[-1] - self.state[2, -1]) / float(self.state[2, -1])
        self.past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        self.past_bandwidths = self.state[2, -5:]
        while self.past_bandwidths[0] == 0.0:
            self.past_bandwidths = self.past_bandwidths[1:]

        bandwidth_sum = 0
        for past_val in self.past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(self.past_bandwidths))


        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if (len(self.past_errors) < 5):
            error_pos = -len(self.past_errors)
        # max_error = float(max(past_errors[error_pos:]))
        error = self.param_function(self.past_errors[error_pos:])
        future_bandwidth = harmonic_bandwidth / (1 + error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - self.state[5, -1] * CHUNK_TIL_VIDEO_END_CAP)
        # print  "state:", state[-1, 5]
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (TOTAL_VIDEO_CHUNKS - last_index < 5):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index


        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = self.state[1, -1] * BUFFER_NORM_FACTOR
        # start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int(np.argwhere(VIDEO_BIT_RATE == self.state[0, -1] * max(VIDEO_BIT_RATE))[0])
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (self.get_chunk_size(chunk_quality,
                                                index) / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += get_qoe_mapping(self.qoe_type, chunk_quality)
                smoothness_diffs += abs(get_qoe_mapping(self.qoe_type, chunk_quality) - get_qoe_mapping(self.qoe_type, last_quality))
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            reward = bitrate_sum - (REBUF_PENALTY * curr_rebuffer_time) - smoothness_diffs
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)

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


