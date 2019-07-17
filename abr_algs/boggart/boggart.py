from constants import *
from abr_algs.abr import ABR
from abr_algs.boggart import contextual_exp3
from abr_algs.boggart.contexts import context_factory


class Boggart(ABR):

    def __init__(self, video, reward, boggart_file=None, context_name=BOGGART_CONTEXT, save_file=None):
        super(Boggart, self).__init__(video, reward)
        self.context_generator = context_factory(self.video, context_name)
        self.trainer = contextual_exp3.Contextual_Exp3(self.context_generator.get_dimension(),
                                                       len(self.context_generator.get_predictions()), boggart_file,
                                                       save_file)

    def get_quality(self, network_state):
        context = self.context_generator.get_context(network_state)
        prediction = self.trainer.predict(context)
        last_quality = network_state[LAST_BR_IDX, -1]
        quality = self.context_generator.get_quality(last_quality, prediction)
        return int(quality)

    def update(self, reward):
        self.trainer.update(reward)

    def save_params(self):
        self.trainer.save()
