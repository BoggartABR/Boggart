from constants import *
from abr_algs.abr import ABR
from enum import Enum
from abr_algs.throughput import quality_from_throughput
import numpy as np



class BolaEnh(ABR):
    minimum_buffer = 10000
    minimum_buffer_per_level = 2000
    low_buffer_safety_factor = 0.5
    low_buffer_safety_factor_init = 0.9

    class State(Enum):
        STARTUP = 1
        STEADY = 2

    def __init__(self, video, reward):
        super(BolaEnh, self).__init__(video, reward)
        config_buffer_size = BUFFER_CAPACITY * M_IN_K
        self.abr_osc = False
        self.no_ibr = False

        utility_offset = 1 - np.log(self.video[BITRATE_LEVELS][0])   # so utilities[0] = 1
        self.utilities = [np.log(b) + utility_offset for b in self.video[BITRATE_LEVELS]]

        buffer = BolaEnh.minimum_buffer
        buffer += BolaEnh.minimum_buffer_per_level * len(self.video[BITRATE_LEVELS])
        buffer = max(buffer, config_buffer_size)
        self.gp = (self.utilities[-1] - 1) / (buffer / BolaEnh.minimum_buffer - 1)
        self.Vp = BolaEnh.minimum_buffer / self.gp

        self.state = BolaEnh.State.STARTUP
        self.placeholder = 0
        self.last_quality = 0

    def quality_from_buffer(self, buffer_level):
        quality = 0
        score = None
        for q in range(len(self.video[BITRATE_LEVELS])):
            s = ((self.Vp * (self.utilities[q] + self.gp) - buffer_level) / self.video[BITRATE_LEVELS][q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def quality_from_buffer_placeholder(self, buffer_level):
        return self.quality_from_buffer(buffer_level + self.placeholder)

    def min_buffer_for_quality(self, quality):

        bitrate = self.video[BITRATE_LEVELS][quality]
        utility = self.utilities[quality]

        level = 0
        for q in range(quality):
            # for each bitrates[q] less than bitrates[quality],
            # BOLA should prefer bitrates[quality]
            # (unless bitrates[q] has higher utility)
            if self.utilities[q] < self.utilities[quality]:
                b = self.video[BITRATE_LEVELS][q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + (bitrate * u - b * utility) / (bitrate - b))
                level = max(level, l)
        return level

    def max_buffer_for_quality(self, quality):
        return self.Vp * (self.utilities[quality] + self.gp)

    def get_quality(self, state):


        throughput = state[THROUGHPUT_IDX, -1] * 8
        buffer_level = state[BUFFER_IDX, -1] * 1000.0

        if self.state == BolaEnh.State.STARTUP:
            if throughput == None:
                return (self.last_quality, 0)
            self.state = BolaEnh.State.STEADY
            self.ibr_safety = BolaEnh.low_buffer_safety_factor_init
            quality = quality_from_throughput(throughput, self.video[VIDEO_CHUNK_LENGTH] * M_IN_K, self.video[BITRATE_LEVELS])
            self.placeholder = self.min_buffer_for_quality(quality) - buffer_level
            self.placeholder = max(0, self.placeholder)
            return quality

        quality = self.quality_from_buffer_placeholder(buffer_level)
        quality_t = quality_from_throughput(throughput, self.video[VIDEO_CHUNK_LENGTH] * M_IN_K, self.video[BITRATE_LEVELS])
        if quality > self.last_quality and quality > quality_t:
            quality = max(self.last_quality, quality_t)
            if not self.abr_osc:
                quality += 1

        max_level = self.max_buffer_for_quality(quality)

        ################
        if quality > 0:
            q = quality
            u = self.utilities[q]
            qq = q - 1
            uu = self.utilities[qq]
            #max_level = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
        ################

        delay = buffer_level + self.placeholder - max_level
        if delay > 0:
            if delay <= self.placeholder:
                self.placeholder -= delay
                delay = 0
            else:
                delay -= self.placeholder
                self.placeholder = 0
        else:
            delay = 0

        if quality == len(self.video[BITRATE_LEVELS]) - 1:
            delay = 0

        # insufficient buffer rule
        if not self.no_ibr:
            safe_size = self.ibr_safety * buffer_level * throughput
            self.ibr_safety *= BolaEnh.low_buffer_safety_factor_init
            self.ibr_safety = max(self.ibr_safety, BolaEnh.low_buffer_safety_factor)
            for q in range(quality):
                if self.video[BITRATE_LEVELS][q + 1] * 4000.0 > safe_size:

                    quality = q
                    delay = 0
                    min_level = self.min_buffer_for_quality(quality)
                    max_placeholder = max(0, min_level - buffer_level)
                    self.placeholder = min(max_placeholder, self.placeholder)
                    break

        return quality

    def update(self, reward):
        return
