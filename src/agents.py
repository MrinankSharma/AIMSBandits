import numpy as np

class ExploreExploit():
    # a explore-exploit bandit.
    def __init__(self, bandit, N, T):
        self.bandit = bandit
        self.N = N
        self.T = T

        if self.bandit.K * self.N > T:
            raise ValueError("This is not a valid strategy; you've run out of T!")

    def step(self):
        N_explore = self.bandit.K * self.N
        if self.bandit.gr < N_explore:
            a_t = self.bandit.gr % N_explore
            self.bandit.pull_arm(a_t)
        else:
            means = [arm.compute_mu_hat() for arm in self.bandit.arms]
            means = np.array(means)
            a_t = np.argmax(means)
            r_t = self.bandit.pull_arm(a_t)

        return r_t