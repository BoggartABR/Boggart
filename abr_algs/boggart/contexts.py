from constants import *
from abc import ABCMeta, abstractmethod
import numpy as np


def context_factory(video, name):
    if name == BOGGART_CONTEXT:
        return BoggartContext(video, name)


# abstract class for context generation. Currently only one context type extends this class,
# however architecture supperts addition of different context types.
class Context:
    __metaclass__ = ABCMeta

    def __init__(self, video, name):
        self.video = video
        self.name = name
        self.predictions = self.video[BITRATE_LEVELS]

    @abstractmethod
    def get_context(self, network_state): raise NotImplementedError

    @abstractmethod
    def get_dimension(self): raise NotImplementedError


class BoggartContext(Context):

    def __init__(self, video, name):
        super(BoggartContext, self).__init__(video, name)
        self.predictions = np.array([-(self.video[BR_DIM] - 1), -np.floor(np.sqrt(self.video[BR_DIM] - 1)), -1, 0, 1,
                                     np.floor(np.sqrt(self.video[BR_DIM] - 1)), (self.video[BR_DIM] - 1)]).astype(int)

    def get_context(self, network_state):
        next_sizes = network_state[NEXT_CHUNKS_START_IDX, :self.video[BR_DIM]]
        throughput = network_state[THROUGHPUT_IDX, -1]
        buffer_size = network_state[BUFFER_IDX, -1]
        sizes_in_kb = (np.array(next_sizes) * M_IN_K)
        last_quality = int(network_state[LAST_BR_IDX, -1])

        qualities_idx = np.array(np.minimum(np.maximum(last_quality + self.predictions, 0), self.video[BR_DIM] - 1))
        qualities = (sizes_in_kb / self.video[VIDEO_CHUNK_LENGTH])[qualities_idx]
        tmp = (np.zeros(len(qualities)) + throughput) - qualities
        tmp = tmp[tmp > 0]
        throughput_idx = max(len(tmp) - 1, 0)

        download_time = np.array(sizes_in_kb / throughput)[qualities_idx]
        tmp = (np.zeros(len(qualities_idx)) + buffer_size) - download_time
        tmp = tmp[tmp > 0]
        buffer_idx = max(len(tmp) - 1, 0)

        return throughput_idx, buffer_idx

    def get_dimension(self):
        return len(self.predictions), len(self.predictions)

    def get_predictions(self):
        return self.predictions

    def get_quality(self, last_quality, prediction):
        return max(min(last_quality + self.predictions[prediction], self.video[BR_DIM] - 1), 0)
