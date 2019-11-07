import numpy as np


class BaseAgent(object):

    def __init__(self, bandit, T):
        self.bandit = bandit
        self.T = T

    def terminated(self):
        if self.bandit.gr > self.T:
            return True
        else:
            return False


class ExploreExploit(BaseAgent):
    # a explore-exploit bandit.
    def __init__(self, bandit, N, T):
        super().__init__(bandit, T)
        self.N = N

        if self.bandit.K * self.N > T:
            raise ValueError("This is not a valid strategy; you've run out of T!")

    def step(self):
        N_explore = self.bandit.K * self.N
        if self.terminated():
            print("Run out of moves!")
            return 0

        if self.bandit.gr < N_explore:
            a_t = self.bandit.gr % self.bandit.K
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)
        else:
            means = [arm.compute_mu_hat() for arm in self.bandit.arms]
            means = np.array(means)
            a_t = np.argmax(means)
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)

        return r_t, cumulated_real_regret, a_t


class EpsGreedy(BaseAgent):
    # a explore-exploit agent
    def __init__(self, bandit, eps, T):
        super().__init__(bandit, T)
        self.eps = eps

    def step(self):
        p = np.random.binomial(1, self.eps)

        if self.terminated():
            print("Run out of moves!")
            return 0

        if p == 1:
            a_t = np.random.randint(0, self.bandit.K)
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)
        else:
            means = [arm.compute_mu_hat() for arm in self.bandit.arms]
            means = np.array(means)
            a_t = np.argmax(means)
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)

        return r_t, cumulated_real_regret, a_t


class UCB(BaseAgent):
    def step(self):
        N_explore = self.bandit.K

        if self.terminated():
            print("Run out of moves!")
            return 0

        if self.bandit.gr < N_explore:
            a_t = self.bandit.gr % self.bandit.K
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)
        else:
            means = [arm.compute_mu_hat() for arm in self.bandit.arms]
            r_ts = [2 * np.log(self.T) / arm.n for arm in self.bandit.arms]
            UCBS = [mean + r_t for (mean, r_t) in zip(means, r_ts)]
            a_t = np.argmax(UCBS)
            r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)

        return r_t, cumulated_real_regret, a_t

class Thompson(BaseAgent):
    def step(self):
        N_explore = self.bandit.K

        if self.terminated():
            print("Run out of moves!")
            return 0

        sample_means = np.zeros(len(self.bandit.arms))
        for indx, arm in enumerate(self.bandit.arms):
            sample_means[indx] = arm.sample_expected_reward()

        a_t = np.argmax(sample_means)
        r_t, cumulated_real_regret = self.bandit.pull_arm(a_t)

        return r_t, cumulated_real_regret, a_t
