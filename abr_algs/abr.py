from abc import ABCMeta, abstractmethod

# abstract ABR algorithms class
class ABR:

    __metaclass__ = ABCMeta

    @classmethod
    def __init__(self, video, reward):
        self.video = video

    @abstractmethod
    def get_quality(self, network_state): raise NotImplementedError

    @abstractmethod
    def update(self, reward): raise NotImplementedError

