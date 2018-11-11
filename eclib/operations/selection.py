'''
Abstruct
'''

import random
from operator import itemgetter


def identity(x):
    return x


class RouletteSelection(object):
    def __init__(self, key=itemgetter(0)):
        # default key: fitst item of touple
        self.key = key

    def __call__(self, population):
        pop = list(population)
        fits = [self.key(x) for x in pop]
        wheel = sum(fits) * random.random() # fitness[0]
        for i, fit in enumerate(fits):
            if fit <= 0:
                continue
            wheel -= fit
            if wheel < 0:
                return pop.pop(i), pop
        raise RuntimeError('Error: in roulette')


class TournamentSelection(object):
    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    def __call__(self, population):
        s = len(population)
        k = min(self.ksize, s)
        pop = list(population)
        indices = random.sample(range(s), k)

        # pop = random.sample(population, k)
        index = max(indices, key=pop.__getitem__)
        return pop.pop(index), pop


class TournamentSelectionStrict(object):
    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    def __call__(self, population):
        s = len(population)
        if s <= 1:
            return None, []
        k = min(self.ksize, s)
        pop = list(population)
        indices = random.sample(range(s), k)

        # pop = random.sample(population, k)
        index = max(indices, key=pop.__getitem__)
        return pop[index], [x for i, x in enumerate(pop) if i not in indices]


class TournamentSelectionDCD(object):
    def __init__(self, key=identity):
        self.key = key
        # self.pat = [0, 0, 0, 0, 0]

    def __call__(self, population):
        s = len(population)
        if s <= 1:
            return None, []
        # k = min(self.ksize, s)
        k = 2
        pop = list(population)
        indices = random.sample(range(s), k)

        # pop = random.sample(population, k)

        def ret(i):
            # return pop.pop(indices[i]), pop
            return getpop(i), [x for i, x in enumerate(pop) if i not in indices]
        def getpop(i):
            return pop[indices[i]]

        if getpop(0).dominates(getpop(1)):
            # self.pat[0] += 1
            return ret(0)
        elif getpop(1).dominates(getpop(0)):
            # self.pat[1] += 1
            return ret(1)

        if len(getpop(0)) >= 2 and len(getpop(1)) >= 2:
            # 混雑度比較
            if getpop(0)[1] < getpop(1)[1]:
                # self.pat[2] += 1
                return ret(1)
            elif getpop(0)[1] > getpop(1)[1]:
                # self.pat[3] += 1
                return ret(0)
        # self.pat[4] += 1

        if random.random() <= 0.5:
            return ret(0)
        return ret(1)
