import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_rl'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/good_log_model2.ckpt'
# NN_MODEL = './models/hd_model.ckpt'
# NN_MODEL = './models/lin_train2_126400.ckpt'
# NN_MODEL = './models/hd_train148600.ckpt'
# NN_MODEL = './models/log_train465800.ckpt'

MAX_REBUF = 15
BUFFER_THRESH = 60.0
VIDEO_CHUNCK_LEN = 4000.0
HD_REWARD = [1, 2, 3, 12, 15, 20]
WRITE = False


class Pensieve:

    def __init__(self, qoe):
        self.sess = tf.Session()
        self.actor = a3c.ActorNetwork(self.sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.nn_model = NN_MODEL
        # self.nn_model = './models/pretrain_linear_reward.ckpt'
        self.saver.restore(self.sess, self.nn_model)
        self.state = np.zeros((S_INFO, S_LEN))

    def predict(self, last_quality, throughput, buffer_size, delay, next_sizes, video_chunk_remain, bitrates):

        with tf.Session() as sess:

            self.state = np.roll(self.state, -1, axis=1)

            self.state[0, -1] = bitrates[last_quality] / float(np.max(bitrates))  # last quality
            self.state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            self.state[2, -1] = throughput    # k-byte to ms
            self.state[3, -1] = float(delay) / BUFFER_NORM_FACTOR  # 10 sec
            self.state[4, :A_DIM] = np.array(next_sizes)  # mega byte
            self.state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = self.actor.predict(np.reshape(self.state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            return bit_rate
