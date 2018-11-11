import random
import numpy as np


class PolynomialBounded():
    def __init__(self, rate=0.1, eta=20):
        self.rate = rate
        self.eta = eta

    def __call__(self, gene):
        size = len(gene)
        res = np.array(gene)
        xl, xu = 0, 1

        for i, x in enumerate(gene):
            if random.random() > self.rate:
                continue
            x = gene[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (self.eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(self.eta + 1)
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(self.eta + 1)
                delta_q = 1.0 - val**mut_pow

            y = x + delta_q * (xu - xl)
            y = min(max(y, xl), xu)
            res[i] = y
        return res

# const pmu = (a=20) => {
#   let _eta = a;
#   const fn = gene => {
#     const dst = [];
#     for (let i = 0, l = gene.length; i < l; ++i) {
#       const r = Math.random();
#       dst.push(clamp(gene[i] + (r <= 0.5 ?
#         ((2 * r) **       (1 / (1 + _eta)) - 1) * gene[i] :
#         ((2 * (1 - r)) ** (1 / (1 + _eta)) - 1) * (gene[i] - 1)), 0, 1));
#     }
#     return dst;
#   }
#   fn.eta = function(a) { _eta = a; return this; }
#   return fn;
# }
