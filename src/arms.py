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

class UniformArm(Arm):
    def __init__(self, l, u):
        self.l = l
        self.u = u
        self.mu = (l + u) / 2

    def pull(self, global_round):
        self.n += 1
        r = np.random.uniform(self.l, self.u)
        self.update_history(global_round, self.n, r)

        return r

    def arm_summary_string(self):
        return f"Uniform Arm, R in [{self.l}, {self.u}]"