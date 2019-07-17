import numpy as np
import tensorflow as tf
import a3c
from abr_algs.abr import ABR
from constants import *

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000


class Pensieve(ABR):

    def __init__(self, video, reward, nn_model):
        super(Pensieve, self).__init__(video, reward)
        self.sess = tf.Session()
        self.actor = a3c.ActorNetwork(self.sess,
                                 state_dim=[INPUT_LEN, HISTORY_LEN], action_dim=self.video[BR_DIM],
                                 learning_rate=ACTOR_LR_RATE)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.nn_model = nn_model
        self.saver.restore(self.sess, self.nn_model)

    def get_quality(self, network_state):

        with tf.Session() as sess:

            curr_state = np.zeros((INPUT_LEN, HISTORY_LEN))
            curr_state[LAST_BR_IDX] = np.array(self.video[BITRATE_LEVELS])[network_state[LAST_BR_IDX].astype(int)] / \
                                         self.video[BITRATE_LEVELS][-1]
            curr_state[BUFFER_IDX] = network_state[BUFFER_IDX] / BUFFER_NORM_FACTOR
            curr_state[THROUGHPUT_IDX] = network_state[THROUGHPUT_IDX] / M_IN_K
            curr_state[DELAY_IDX] = network_state[DELAY_IDX]
            curr_state[NEXT_CHUNKS_START_IDX] = network_state[NEXT_CHUNKS_START_IDX]
            curr_state[CHUNKS_TILL_END_IDX] = network_state[CHUNKS_TILL_END_IDX] / float(self.video[NUM_OF_CHUNKS])

            action_prob = self.actor.predict(np.reshape(curr_state, (1, INPUT_LEN, HISTORY_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            return bit_rate

    def update(self, reward):
        return
