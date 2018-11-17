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

        # self.is_dominated = np.empty((size, size), dtype=np.bool)
        # self.num_dominated = np.empty(size, dtype=np.uint32)
        # self.mask = np.empty(size, dtype=np.bool)
        # self.rank = np.empty(size, dtype=np.uint32)

    def __call__(self, population, n=None, return_rank=False):
        # return sortNondominated_v1(population, n=n)

        popsize = len(population)
        if not popsize:
            Exception('Error: population size is 0')

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.empty(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        for i in range(popsize):
            for j in range(popsize):
                # iはjに優越されているか
                isdom = i != j and population[j].dominates(population[i])
                is_dominated[i, j] = isdom

        # iを優越している個体を数える
        is_dominated.sum(axis=(1,), out=num_dominated)

        fronts = []
        lim = popsize if n is None else n
        for r in range(popsize):
            # ランク未決定かつ最前線であるかを判定
            front = []
            for i in range(popsize):
                # ランク未決定又は優越される個体がない->ランク決定
                isrankdetermined = not (rank[i] or num_dominated[i])
                mask[i] = isrankdetermined
                if isrankdetermined:
                    rank[i] = r + 1
                    front.append(population[i])

            fronts.append(front)
            lim -= len(front)

            # 終了判定
            if return_rank:
                if rank.all():
                    return rank
            elif lim <= 0:
                return fronts

            # 優越数更新
            # for i in range(popsize):
            #     num_dominated[i] -= np.sum(mask * is_dominated[i, :])
            # print(mask.dtype)
            # print(is_dominated.dtype)
            # print((mask & is_dominated).sum(axis=(1,)))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception('Error: reached the end of function')


class NondominatedSortIterator(object):
    '''
    def cmp(a, b):
        if a == b: return 0
        return -1 if a < b else 1
    '''
    def __init__(self, population):
        self._population = population
        self._size = len(population)

        if not self._size:
            Exception('Error: population is empty')

    def __iter__(self):
        pop = self._population
        lim = self._size

        is_dominated = np.empty((self._size, self._size), dtype=np.bool)
        num_dominated = np.empty(self._size, dtype=np.int64)
        mask = np.empty(self._size, dtype=np.bool)
        rank = np.zeros(self._size, dtype=np.int64)

        for i in range(self._size):
            for j in range(self._size):
                # iはjに優越されているか
                is_dominated[i, j] = i != j and pop[j].dominates(pop[i])

        # iを優越している個体を数える
        is_dominated.sum(axis=(1,), out=num_dominated)

        for r in range(self._size):
            # ランク未決定かつ最前線であるかを判定
            front = []
            for i in range(self._size):
                # ランク未決定又は優越される個体がない->ランク決定
                isrankdetermined = not (rank[i] or num_dominated[i])
                mask[i] = isrankdetermined
                if isrankdetermined:
                    rank[i] = r + 1
                    front.append(pop[i])
                    lim -= 1
            yield front

            # 終了判定
            if lim <= 0:
                raise StopIteration()

            # 優越数更新
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception('Error: reached the end of function')


################################################################################

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


################################################################################

from collections import defaultdict


def sortNondominated_v1(population, n=None, first_front_only=False):
    k = len(population) if n is None else n
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for fit in population:
        map_fit_ind[fit.data.value].append(fit)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    def dominates(fit0, fit1):
        return map_fit_ind[fit0][0].dominates(map_fit_ind[fit1][0])

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if dominates(fit_i, fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all population are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(population), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts
