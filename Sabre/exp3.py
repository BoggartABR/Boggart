import numpy as np
import scipy.misc as sc
from scipy.special import logsumexp
import copy

CONTEXT = None#(0,4,5)
NORMALIZE_THRESH = 10000
class Exp3:

    def __init__(self, num_of_arms, context, to_record, weights = None, gamma = 0):
        if gamma:
            self.gamma = gamma
        else:
            self.gamma = min(1.0, np.sqrt(np.log(num_of_arms) / num_of_arms))

        if weights:
            self.weights = np.array(weights)
        else:
            self.weights = np.zeros(num_of_arms)
        self.num_of_arms = num_of_arms
        self.last_arm = 0
        self.t = 1
        self.context = context
        if to_record is not None:
            self.to_record = to_record + str(context)
        else:
            self.to_record = None
        self.counter = 0
        self.printed = False
        self.arm_counter = np.zeros(num_of_arms)



    def predict(self):

        if self.gamma != 1.0:
            c = logsumexp(-self.gamma * self.weights)
            prob_dist = np.exp((-self.gamma * self.weights) - c)
        else:
            prob_dist = np.ones(self.num_of_arms) / np.float(self.num_of_arms)
        # print prob_dist
        arm = np.random.choice(a=self.num_of_arms, p=prob_dist)
        self.last_arm = arm
        self.arm_counter[arm] += 1
        return arm


    def update(self, reward, last_arm = None):
        if last_arm == None:
            last_arm = self.last_arm
        loss = 1 - reward

        # if self.context == (1,6) and self.last_arm == 0:
        #     print self.arm_counter
        #     print reward
        #
        # if self.counter >= NORMALIZE_THRESH - 1 and self.context == (1,6):
        #     print ("OH NO!")
        #     print self.last_arm
        #     print self.weights
        #     print reward

        if self.gamma != 1.0:
            c = logsumexp(-self.gamma * self.weights)
            prob = np.exp((-self.gamma * self.weights[last_arm]) - c)
        else:
            prob = 1.0 / np.float(self.num_of_arms)

        # if prob <= 0:
        #     print self.context
        #     print self.weights
        #     print self.last_arm

        assert prob > 0

        weight_holder = copy.deepcopy(self.weights)

        estimated = loss / prob
        self.weights[last_arm] += estimated
        # self.t = max(self.t + 1, 1000)
        self.t += 1
        self.counter += 1
        self.gamma = min(1.0, np.sqrt(np.log(self.num_of_arms) / (self.num_of_arms * self.t)))

        # self.gamma = np.sqrt(np.log(self.num_of_arms) / (self.num_of_arms))
        #
        # if self.counter >= NORMALIZE_THRESH :
        #     # if self.context == (1,6):
        #     #     print "after updating"
        #     #     print self.weights
        #     self.weights -= sc.logsumexp(self.weights)
        #     # print self.weights
        #     self.counter = 0



    def is_more_weight(self, arm, part):
        return (self.weights[arm] / sum(self.weights)) >= part

    def record_weights(self):
        if self.to_record is None:
            return
        if self.gamma != 1.0:
            c = logsumexp(-self.gamma * self.weights)
            prob_dist = np.exp((-self.gamma * self.weights) - c)
        else:
            prob_dist = np.ones(self.num_of_arms) / np.float(self.num_of_arms)
        with open(self.to_record, 'a') as f:
            f.write(str(np.around(prob_dist, 2)))
            f.write('\n')


    def entropy_reward(self, activity_steps_threshold):
        if self.t < activity_steps_threshold:
            return -1, -1
        c = logsumexp(-self.gamma * self.weights)
        prob_dist = np.exp((-self.gamma * self.weights) - c)

        # entropy close to 0 is good, so we want the peak to be at 0, as in cosinus.
        # TODO: change normalization
        prob_dist = prob_dist + 1e-80
        return 1 - np.sum(prob_dist * np.log(prob_dist)), prob_dist > 0.8