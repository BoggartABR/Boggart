from abr_algs.abr import ABR
from constants import *


RESEVOIR = 5
CUSHION = 10

class BBA(ABR):

    def get_quality(self, network_state):
        buffer_size = network_state[BUFFER_IDX, -1]
        # buffer_size = last_state[BUFFER_IDX]

        if buffer_size < RESEVOIR:
            quality = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            quality = len(self.video[BITRATE_LEVELS]) - 1
        else:
            quality = (len(self.video[BITRATE_LEVELS]) - 1) * (buffer_size - RESEVOIR) / float(CUSHION)
        return int(quality)

    def update(self, reward):
        return