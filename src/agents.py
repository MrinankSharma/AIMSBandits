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
    def __init__(self, bandit, N, T, opt=False):
        super().__init__(bandit, T)
        self.N = N

        if opt == True:
            self.N = np.floor((T ** (2.0 / 3)) * (np.log(T) ** (1.0 / 3)))

        if self.bandit.K * self.N > T:
            print("Warning: weird setting of N, using a more sensible value!")
            self.N = np.floor((T/2*N))

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


class OptimalEpsGreedy(EpsGreedy):

    def step(self):
        if self.bandit.gr == 0:
            # explore on the first round.
            self.eps = 1
        else:
            self.eps = (self.bandit.gr ** (-1/3)) * (self.bandit.K * np.log(self.bandit.gr))**(1/3)

        if self.eps > 1:
            self.eps = 1
        elif self.eps < 0:
            self.eps = 0
        
        return super(OptimalEpsGreedy, self).step()

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
            r_ts = [arm.compute_rt(self.T) for arm in self.bandit.arms]
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
