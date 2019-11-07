import numpy as np

from abc import ABC, abstractmethod
class Arm(ABC):

    def __init__(self):
        self.mu = 0
        self.n = 0

        self.history = {
            "global_round": [],
            "n": [],
            "r": [],
        }

    def update_history(self, round, n, reward):
        self.history["global_round"].append(round)
        self.history["n"].append(self.n)
        self.history["r"].append(reward)

    def compute_mu_hat(self):
        rs = np.array(self.history["r"])
        mean = np.mean(rs)
        return mean

    def reset(self):
        self.n = 0
        self.history = {
            "global_round": [],
            "n": [],
            "r": [],
        }

    def compute_rt(self, T):
        return (2.0 * np.log(T) / self.n)**0.5

class UniformArm(Arm):

    @classmethod
    def factory(cls, m = 0.5):
        def factory_function(m):
            l = np.random.uniform(0, m)
            u = np.random.uniform(l, 1)
            return UniformArm(l, u)
        return lambda: factory_function(m)

    def __init__(self, l, u):
        super().__init__()
        self.l = l
        self.u = u
        self.mu = (l + u) / 2

    def pull(self, global_round):
        self.n += 1
        r = np.random.uniform(self.l, self.u)
        self.update_history(global_round, self.n, r)

        return r

    def arm_summary_string(self):
        return f"Uniform Arm mu {self.mu}, R in [{self.l}, {self.u}]"


class GaussianArm(Arm):

    @classmethod
    def factory(cls, mu_p = 0.5, sigma_p = 1, sigma_n = 0.1):
        def factory_function():
            mu = np.random.normal(mu_p, sigma_p)
            return GaussianArm(mu, sigma_n, mu_p, sigma_p)
        return lambda: factory_function()

    def __init__(self, mu, sigma, mu_p, sigma_p):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.prior_mu = mu_p
        self.prior_sigma = sigma_p
        self.post_mu = mu_p
        self.post_sigma = sigma_p

    def pull(self, global_round):
        self.n += 1
        r = np.random.normal(self.mu, self.sigma)
        self.update_history(global_round, self.n, r)
        self.update_posterior(r)

        return r

    def arm_summary_string(self):
        return f"Gaussian Arm mu {self.mu}, sigma {self.sigma}.\nPosterior is {self.post_mu}, {self.post_sigma}\n"

    def update_posterior(self, r):
        post_pres = 1/(self.post_sigma**2)
        obs_pres = 1/(self.sigma ** 2)
        post_nat_mean = post_pres * self.post_mu
        obs_nat_mean = obs_pres * r

        post_nat_mean = obs_nat_mean + post_nat_mean
        post_pres = obs_pres + post_pres
        post_sigma = 1/(post_pres**0.5)
        post_mu = post_nat_mean / post_pres

        # the posterior becomes the new post
        self.post_mu = post_mu
        self.post_sigma = post_sigma
        return post_mu, post_sigma

    def sample_expected_reward(self):
        return np.random.normal(self.post_mu, self.post_sigma)

    def reset(self):
        super().reset()
        self.post_mu = self.prior_mu
        self.post_sigma = self.prior_sigma

    def compute_rt(self, T):
        return self.sigma * (8 * np.log(T) / self.n)**0.5

class BernoulliArm(Arm):

    @classmethod
    def factory(cls, alpha, beta):
        def factory_function():
            mu = np.random.beta(alpha, beta)
            return BernoulliArm(mu, alpha, beta)
        return lambda: factory_function()

    def __init__(self, mu, alpha, beta):
        super().__init__()
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.S = 0
        self.F = 0

    def pull(self, global_round):
        self.n += 1
        r = np.random.binomial(1, self.mu)
        self.update_history(global_round, self.n, r)
        self.update_posterior(r)

        return r

    def arm_summary_string(self):
        return f"Bernoulli Arm mu {self.mu}, \nPosterior is {self.S + self.alpha}, {self.F + self.beta}\n"

    def update_posterior(self, r):
        if r == 0:
            self.F += 1
        else:
            self.S += 1

        return self.S + self.alpha, self.F + self.beta

    def sample_expected_reward(self):
        return np.random.beta(self.S + self.alpha, self.F + self.beta)

    def reset(self):
        super().reset()
        self.S = 0
        self.F = 0
