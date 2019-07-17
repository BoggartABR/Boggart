from constants import *
from abr_algs.abr import ABR

SAFETY_FACTOR = 0.9


def quality_from_throughput(tput, chunk_length, bitrates):
    quality = 0
    while (quality + 1 < len(bitrates) and
           chunk_length * bitrates[quality + 1] / tput <= chunk_length):
        quality += 1

    return quality


class Throughput(ABR):

    def get_quality(self, network_state):

        throughput_hist = network_state[THROUGHPUT_IDX, :] * BITS_IN_BYTE
        # if beginning of video, trim history
        while throughput_hist[0] == 0:
            throughput_hist = throughput_hist[1:]
        # throughput_hist = throughput_hist * BITS_IN_BYTE * M_IN_K  # in kbit ps

        return quality_from_throughput((sum(throughput_hist) / len(throughput_hist)) * SAFETY_FACTOR,
                                       self.video[VIDEO_CHUNK_LENGTH] * M_IN_K,
                                       self.video[BITRATE_LEVELS])

    def update(self, reward):
        return
