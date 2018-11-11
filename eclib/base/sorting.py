'''
'''

import numpy as np


def identity(x):
    return x


class NondominatedSort(object):
    '''
    def cmp(a, b):
        if a == b: return 0
        return -1 if a < b else 1
    '''
    def __init__(self, size=100):
        self.size = size

        self.is_dominated = np.empty((size, size), dtype=np.uint8)
        self.num_dominated = np.empty(size, dtype=np.uint32)
        self.mask = np.empty(size, dtype=np.uint8)
        self.rank = np.empty(size, dtype=np.uint32)

    def __call__(self, population):
        popsize = len(population)
        if not popsize:
            Exception('Error: population size is 0')

        if popsize != self.size:
            self.size = popsize
            self.is_dominated = np.empty((popsize, popsize), dtype=np.uint32)
            self.num_dominated = np.empty(popsize, dtype=np.uint32)
            self.mask = np.empty(popsize, dtype=np.uint32)
            # self.rank = np.zeros(popsize, dtype=np.uint32)
        self.rank = np.zeros(popsize, dtype=np.uint32)

        for i in range(popsize):
            for j in range(popsize):
                # iはjに優越されているか
                if i != j and population[j].dominates(population[i]):
                    self.is_dominated[i, j] = 1
                else:
                    self.is_dominated[i, j] = 0

        # iを優越している個体を数える
        self.is_dominated.sum(axis=(1,), out=self.num_dominated)

        fronts = []
        for r in range(popsize):
            # ランク未決定かつ最前線であるかを判定
            front = []
            for i in range(popsize):
                if self.rank[i] or self.num_dominated[i]:
                    # ランク決定済み又は優越個体がある
                    self.mask[i] = 0
                else:
                    self.mask[i] = 1

                if self.mask[i]:
                    self.rank[i] = r + 1
                    front.append(population[i])

            fronts.append(front)

            # 終了判定
            if self.rank.all():
                return fronts
                # return self.rank

            # 優越数更新
            for i in range(popsize):
                self.num_dominated[i] -= (self.mask * self.is_dominated[i, :]).sum()

        raise Exception('Error: reached the end of function')


class CrowdingDistanceCalculator(object):
    def __init__(self, key=identity):
        self.key = key

    def __call__(self, population):
        popsize = len(population)
        if popsize == 0:
            return

        distances = np.zeros(popsize, dtype=np.float32)

        values = [self.key(x) for x in population]
        index = list(range(popsize))

        nobj = len(values[0])

        for i in range(nobj):
            get_value = lambda idx: values[idx][i]

            index.sort(key=get_value)

            distances[index[0]] = float('Infinity')
            distances[index[-1]] = float('Infinity')

            vrange = get_value(-1) - get_value(0)

            if vrange <= 0:
                continue

            norm = nobj * vrange

            for l, c, r in zip(index[:-2], index[1:-1], index[2:]):
                distances[c] += (get_value(r) - get_value(l)) / norm

        return distances
