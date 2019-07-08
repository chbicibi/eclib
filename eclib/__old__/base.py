'''
'''

import numpy as np


# const compareFn = [
#   (self, other) => {
#     for (i = 0, l = self.value.length i < l ++i)
#       if (self.value[i] < other.value[i]) return false
#     for (i = 0, l = self.value.length i < l ++i)
#       if (self.value[i] != other.value[i]) return true
#     return false
#   },
#   (self, other) => {
#     for (i = 0, l = self.value.length i < l ++i)
#       if (self.value[i] <= other.value[i]) return false
#     return true
#   }
# ] # 0 => weak, 1 => strong

class NondominatedSort(object):
    '''
    def cmp(a, b):
        if a == b: return 0
        return -1 if a < b else 1
    '''

    def __init__(self, cmp, size=100):
        self.cmp = cmp
        self.size = size

        self.is_dominated = np.empty((size, size), dtype=np.uint8)
        self.num_dominated = np.empty(size, dtype=np.uint32)
        self.mask = np.empty(size, dtype=np.uint8)
        self.rank = np.empty(size, dtype=np.uint32)

    def __call__(self, population):
        popsize = len(population)

        if popsize != self.size:
            self.size = popsize
            self.is_dominated = np.empty((popsize, popsize), dtype=np.uint32)
            self.num_dominated = np.empty(popsize, dtype=np.uint32)
            self.mask = np.empty(popsize, dtype=np.uint32)
            # self.rank = np.empty(popsize, dtype=np.uint32)
        if True:
            self.rank = np.zeros(popsize, dtype=np.uint32)

        for i in range(popsize):
            for j in range(popsize):
                # iはjに優越されているか
                if i != j and self.isdom(population[i], population[j]):
                    self.is_dominated[i, j] = 1#i !== j && fn(pop[i], pop[j])
                else:
                    self.is_dominated[i, j] = 0

            # iを優越している個体を数える
            self.is_dominated.sum(axis=(1,), out=self.num_dominated)

        for r in range(popsize):
            # ランク未決定かつ最前線であるかを判定
            for i in range(popsize):
                self.mask[i] = 0 if self.rank[i] or self.num_dominated[i] else 1
                if self.mask[i]:
                    self.rank[i] = r + 1

            # 終了判定
            if self.rank.all():
                return self.callback(self.rank)

            # 優越数更新
            for i in range(popsize):
                self.num_dominated[i] -= (self.mask * self.is_dominated[i, :]).sum()

        raise RuntimeError('Error: in ranking')

    def isdom(self, a, b):
        return np.all(a() >= b()) and np.any(a() != b())

    def callback(self, rank):
        return rank

#   fn.cpFn = function(f) { isDom = f return this }
#   fn.setFn = function(f) { setFn = f return this }
#   return fn
# }

# const crowding = (pop, idx=[0, 1]) => {
#   const scale = idx.map(i => {
#     const range = d3.extent(pop, ind => ind.value[i])
#     return v => range[1] === range[0] ? 0 : (Math.abs(v) / (range[1] - range[0]))
#   })
#   const l = pop.length
#   const a = d3.range(l).sort((i, j) => pop[i].value[idx[0]] - pop[j].value[idx[0]])
#   const d = new Array(a.length)
#   d[a[0]] = d[a[l - 1]] = 1
#   for (i = 1 i < l - 1 ++i) {
#     d[a[i]] = d3.sum(idx, (j, k) => scale[k](pop[a[i + 1]].value[j] - pop[a[i - 1]].value[j]))
#   }
#   return d
# }
