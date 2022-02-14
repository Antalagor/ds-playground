# Returns an advantage weight tensor that can be used for policy optimization.
# This can be used as target for advantage estimation (a value net that depends on observations) or fed in directly.

import numpy as np


class ValueTargetCalculator:
    def calculate_new_values(self, rewards, dones, values):
        """
        :param rewards: reward from env actions
        :param dones: done from env action
        :param values: value from obs value prediction
        :return: value_target, q_value estimate respectively
        """
        raise NotImplementedError


class GeneralizedAdvantageEstimator(ValueTargetCalculator):
    def __init__(self, gam=0.99, lam=0.95):
        self.lam = lam
        self.gam = gam

    def calculate_new_values(self, rewards, dones, values):
        n = len(rewards)
        advantage = np.zeros((len(rewards, )))

        next_advantage = next_value = 0
        for i in reversed(range(n)):
            delta = - values[i] + rewards[i] + (not dones[i]) * self.gam * next_value
            advantage[i] = delta + (not dones[i]) * self.lam * self.gam * next_advantage
            next_value = values[i]
            next_advantage = advantage[i]

        return advantage + values


class AccumForwardRewards(GeneralizedAdvantageEstimator):
    def __init__(self, discount=0.99):
        super(AccumForwardRewards, self).__init__(gam=discount, lam=1)


class TemporalDifference(GeneralizedAdvantageEstimator):
    def __init__(self, discount=0.99):
        super(TemporalDifference, self).__init__(gam=discount, lam=0)
