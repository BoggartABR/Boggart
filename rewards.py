from constants import *
from abc import ABCMeta, abstractmethod
import numpy as np


def get_reward_class(video, type, smooth_penalty = SMOOTH_PENALTY):

    if type == LIN:
        return LinReward(video, type, smooth_penalty)
    if type == HD:
        return HD_REWARD(video, type, smooth_penalty)
    if type == LOG:
        return LogReward(video, type, smooth_penalty)


class Reward():
    __metaclass__ = ABCMeta

    @classmethod
    def __init__(self, video, type, smooth_penalty = SMOOTH_PENALTY):
        self.type = type
        self.video = video
        self.smooth_penalty = smooth_penalty

    # normalized reward used to train Boggart
    def norm_reward(self, quality, last_quality, rebuf):
        min_reward = self.mapping(min(self.video[BITRATE_LEVELS])) \
                     - self.rebuf_penalty * MAX_REBUF \
                     - self.smooth_penalty \
                     * (self.mapping(max(self.video[BITRATE_LEVELS])) - self.mapping(min(self.video[BITRATE_LEVELS])))

        max_reward = self.mapping(max(self.video[BITRATE_LEVELS]))
        return (self.reward(quality, last_quality, rebuf) - min_reward) / (max_reward - min_reward)

    def reward(self, quality, last_quality, rebuf):
        reward = self.mapping(self.video[BITRATE_LEVELS][quality]) \
                 - self.rebuf_penalty * min(rebuf, MAX_REBUF) \
                 - self.smooth_penalty * np.abs(self.mapping(self.video[BITRATE_LEVELS][quality])
                                                - self.mapping(self.video[BITRATE_LEVELS][last_quality]))
        return reward


class LinReward(Reward):

    def __init__(self, video, type, rebuf_penalty = LIN_REBUF_PENALTY, smooth_penalty = SMOOTH_PENALTY):
        super(LinReward, self).__init__( video, type, smooth_penalty)
        self.rebuf_penalty = rebuf_penalty

    def mapping(self, bitrate):
        return bitrate / M_IN_K

class HDReward(Reward):

    def __init__(self, video, type, rebuf_penalty = HD_REBUF_PENALTY, smooth_penalty = SMOOTH_PENALTY):
        super(HDReward, self).__init__( video, type, smooth_penalty)
        self.rebuf_penalty = rebuf_penalty

    def mapping(self, bitrate):
        return self.video[HD_REWARD][np.argwhere(self.video[BITRATE_LEVELS] == bitrate)[0][0]]

class LogReward(Reward):

    def __init__(self, video, type, rebuf_penalty =  LOG_REBUF_PENALTY, smooth_penalty = SMOOTH_PENALTY):
        super(LogReward, self).__init__( video, type, smooth_penalty)
        self.rebuf_penalty = rebuf_penalty

    def mapping(self, bitrate):
        return np.log(bitrate / float(self.video[BITRATE_LEVELS][0]))

