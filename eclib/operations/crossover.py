import random
import numpy as np


class BLXCrossover(object):
    def __init__(self, rate=0.9, alpha=0.5, oneout=False):
        self.rate = rate
        self.alpha = alpha
        self.oneout = oneout

    def __call__(self, origin):
        x1 = origin[0].get_gene()
        x2 = origin[1].get_gene()

        if random.random() > self.rate:
            return x1, x2

        gamma = (1 + 2 * self.alpha) * np.random.random(x1.shape) - self.alpha

        if self.oneout:
            y = (1 - gamma) * x1 + gamma * x2
            return y
        else:
            y1 = (1 - gamma) * x1 + gamma * x2
            y2 = gamma * x1 + (1 - gamma) * x2
            return y1, y2


class SimulatedBinaryBounded(object):
    def __init__(self, rate=0.9, eta=20, oneout=False):
        self.rate = rate
        self.eta = eta
        self.oneout = oneout

    def __call__(self, origin):
        # x1 = origin[0].get_gene()
        # x2 = origin[1].get_gene()

        y1, y2 = (np.array(x.get_gene()) for x in origin[:2])

        if random.random() > self.rate:
            return y1, y2

        size = min(len(y1), len(y2))

        xl, xu = 0, 1
        eta = self.eta

        for i in range(size):
            if random.random() <= 0.5:
                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                if abs(y1[i] - y2[i]) > 1e-14:
                    x1 = min(y1[i], y2[i])
                    x2 = max(y1[i], y2[i])
                    rand = random.random()

                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                    c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)

                    if random.random() <= 0.5:
                        y1[i] = c2
                        y2[i] = c1
                    else:
                        y1[i] = c1
                        y2[i] = c2

        if self.oneout:
            return y1
        else:
            return y1, y2
