from abr_algs.boggart import exp3
import json
import os

def calc_inner_index(dim, idx):
    assert len(dim) == len(idx)
    if len(dim) == 0:
        return 0
    tmp = 1
    for i in dim[1:]:
        tmp *= i
    return idx[0] * tmp + calc_inner_index(dim[1:], idx[1:])

# Comtextual-MAB class that holds the list of all separately trained exp3 algorithm instances. Essentially a
# demultiplexer that receives a context and updates / outputs quality according to suitable exp3 algorithm
# instance.

class Contextual_Exp3:

    def __init__(self, context_dim, num_of_arms, boggart_file, to_save = None):

        self.num_of_arms = num_of_arms      # number of actions of MAB algorithm
        self.context_dim = context_dim      # dimension of the context
        self.flat_dim = 1
        for i in context_dim:
            self.flat_dim *= i

        self.exp3_list = [None] * self.flat_dim

        # loads trained Boggart model if one os given
        if boggart_file is not None:
            gammas_file = boggart_file + '_gamma.json'
            weights_file = boggart_file + '_weights.json'
            with open(weights_file) as wf:
                weights = json.load(wf)
            with open(gammas_file) as gf:
                gammas = json.load(gf)
            for context in weights.keys():
                weight = weights[context]
                gamma = gammas[context]
                context = [int(i) for i in context.split()]

                flat_idx = calc_inner_index(self.context_dim, context)
                self.exp3_list[flat_idx] = exp3.Exp3(self.num_of_arms, context, weight, gamma)

        self.first_round = False
        self.last_exp3 = None       # needs to store last active exp3 instance for updating is in training mode
        self.last_context = None
        self.to_save = to_save      # file name to save model if training

    # return predicted action from exp3 instance according to given context
    def predict(self, context):

        flat_idx = calc_inner_index(self.context_dim, context)

        # if encountered previously unseen context
        if self.exp3_list[flat_idx] is None:
            self.exp3_list[flat_idx] = exp3.Exp3(self.num_of_arms, context)
            self.first_round = True
        else:
            self.first_round = False
        self.last_exp3 = self.exp3_list[flat_idx]
        self.last_context = context

        # get prediction from exp3 instance
        arm = self.exp3_list[flat_idx].predict()
        return arm

    # update weights of last active exp3 instance according to received reward
    def update(self, reward):
        self.last_exp3.update(reward)

    # save boggart params if in training mode
    def save(self):
        if self.to_save is None:
            return

        weights_dict = {}
        gammas_dict = {}
        for alg in self.exp3_list:
            if alg is not None:
                context = alg.context
                weights_dict[context] = alg.weights
                gammas_dict[context] = alg.gamma

        if os.path.isfile(self.to_save + '_weights.json'):
            os.remove(self.to_save + '_weights.json')
        if os.path.isfile(self.to_save + '_gamma.json'):
            os.remove(self.to_save + '_gamma.json')
        with open(self.to_save + '_weights.json', 'wb') as f:
            json.dump(weights_dict, f)
        with open(self.to_save + '_gamma.json', 'wb') as f:
            json.dump(gammas_dict, f)
