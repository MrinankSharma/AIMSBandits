import numpy as np


class BaseStochasticBandit(object):
    def __init__(self, arms, print_sum=True):
        self.arms = arms
        self.mu_star = 0
        self.a_star = 0
        self.K = len(arms)
        self.real_regret = []
        self.rewards = []

        for i in range(self.K):
            if self.arms[i].mu > self.mu_star:
                self.a_star = i
                self.mu_star = self.arms[i].mu

        self.gr = 0

        if print_sum:
            print(self.summary_str())

    def reset(self, print_sum=True):
        self.real_regret = []
        self.rewards = []
        [arm.reset() for arm in self.arms]
        self.gr = 0

        if print_sum:
            print(self.summary_str())

    def pull_arm(self, a_t):
        if a_t < 0 or a_t > (self.K - 1):
            raise ValueError("Trying to Pull Invalid Arm")

        r_t = self.arms[a_t].pull(self.gr)
        # by definition > 0
        delta_t = self.mu_star - self.arms[a_t].mu

        if len(self.real_regret) == 0:
            c_real_regret = delta_t
        else:
            c_real_regret = self.real_regret[-1] + delta_t

        self.real_regret.append(c_real_regret)
        self.gr = self.gr + 1
        self.rewards.append(r_t)
        return r_t, c_real_regret

    def summary_str(self):
        str = f"Multi-Armed Bandit Problem\n K={self.K}\n mu_star = {self.mu_star}\n a_star  = {self.a_star}\n"
        for arm in self.arms:
            str = str + f"\n{arm.arm_summary_string()}"
        return str


class RandomizedStochasticBandit(BaseStochasticBandit):
    # generate a new bandit problem when specifying the base class for each arm and a function generating parameters for
    # the bandit

    def __init__(self, K, arm_generator):
        arms = [arm_generator() for i in range(K)]
        super().__init__(arms)
