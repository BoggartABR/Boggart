import exp3
import pickle
import numpy as np


A_DIM = 6


def calc_inner_index(dim, idx):
    # print("dim:", len(dim), "idx:", len(idx))
    assert len(dim) == len(idx)
    if len(dim) == 0:
        return 0
    tmp = 1
    for i in dim[1:]:
        tmp *= i
    return idx[0] * tmp + calc_inner_index(dim[1:], idx[1:])

def function():
    return 2


def kl_divergence(p, q):
    q = q + 1e-90
    p = p + 1e-90
    return - np.dot(p, np.log(p / q))


class Contextual_Exp3:

    def __init__(self, context_dim, num_of_arms, to_save = None, save_contexts = None):
        self.num_of_arms = num_of_arms
        self.context_dim = context_dim
        self.flat_dim = 1
        for i in context_dim:
            self.flat_dim *= i
        self.exp3_list = [None] * self.flat_dim
        self.first_round = False
        self.last_exp3 = None
        self.last_context = None
        self.to_save = to_save
        self.arms_counter = []
        self.arms_counter.append(np.zeros((self.flat_dim, A_DIM)))
        self.contexts_counter = np.zeros(self.flat_dim)
        self.save_contexts = save_contexts
        # self.const_arm = False
        # self.best_arms = np.zeros(self.flat_dim)

    def load_weights(self, weights, gammas):
        for context in weights.keys():
            gamma = gammas[context]
            weight = weights[context]
            context = [int(i) for i in context.split()]

            flat_idx = calc_inner_index(self.context_dim, context)
            self.exp3_list[flat_idx] = exp3.Exp3(self.num_of_arms, context, self.save_contexts, weight, gamma)



    def predict(self, context):

        flat_idx = calc_inner_index(self.context_dim, context)

        # if self.const_arm:
        #     return np.int(self.best_arms[flat_idx])

        self.contexts_counter[flat_idx] += 1

        if self.exp3_list[flat_idx] is None:
            self.exp3_list[flat_idx] = exp3.Exp3(self.num_of_arms, context, self.save_contexts)
            self.first_round = True
        else:
            self.first_round = False
        self.last_exp3 = self.exp3_list[flat_idx]
        self.last_context = context
        arm = self.exp3_list[flat_idx].predict()
        # self.arms_counter[-1][flat_idx, arm] += 1
        return arm

    def update(self, reward, first_round, chunks_till_end):
        # if self.const_arm:
        #     return
        if first_round:
            return
        self.last_exp3.update(reward)
