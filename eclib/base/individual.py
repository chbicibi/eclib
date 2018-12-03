# import random
from functools import total_ordering
import numpy as np
# from .collections import Container


class IndividualNotEvaluated(Exception):
    pass


################################################################################

class Genome(np.ndarray):
    ''' GA遺伝子を格納するデータ構造
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class Individual(object):
    ''' 進化計算個体
    遺伝子と評価値を管理する
    '''
    # current_id = 0
    # pool = []
    bounds = None # None or (low, up) or ([low], [up])
    weight = None # 重み(正=>最小化, 負=>最大化)

    def __init__(self, genome, origin=None):
        self.genome = genome # 遺伝子型
        self.origin = origin # 派生元 (初期化関数又は関数と引数の組)
        self.value = None    # 評価値 (デフォルトではシーケンス型)
        self.wvalue = None   # 重み付き評価値

        # self.id = Individual.current_id
        # Individual.current_id += 1
        # self.id = len(type(self).pool)
        # type(self).current_id = self.id + 1
        # type(self).pool.append(self)

    def __getitem__(self, key):
        if self.value is None:
            raise IndividualNotEvaluated()
        return self.value[key]

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return f'indiv{self.id}'

    # 以下の6つの比較用メソッドは優越関係を表すものでそれぞれ対になるメソッドの否定ではない
    # self < other != not self <= other
    def __eq__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s == o for s, o in zip(self.wvalue, other.wvalue))

    def __ne__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return any(s != o for s, o in zip(self.wvalue, other.wvalue))

    def __lt__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s < o for s, o in zip(self.wvalue, other.wvalue))

    def __le__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s <= o for s, o in zip(self.wvalue, other.wvalue))

    def __gt__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s > o for s, o in zip(self.wvalue, other.wvalue))

    def __ge__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s >= o for s, o in zip(self.wvalue, other.wvalue))

    def evaluate(self, function):
        if not self.evaluated():
            self.function = function
            self.value = function(self.get_variable())
            if self.weight is not None:
                self.wvalue = self.weight * self.value
            else:
                self.wvalue = self.value
        return Fitness(self)

    def get_variable(self):
        ''' getter of phenotype '''
        return self.decode(self.genome)

    def evaluated(self):
        return self.value is not None

    def get_gene(self, *args, **kwargs):
        ''' getter of genotype
        [memo]
            gene: a part of genome (single type), 遺伝子
            genome: 1個体の完全な遺伝情報, ゲノム
            ここでは gene == genome
        '''
        return self.genome

    # def encode(self, x):
    #     ''' phenotype -> genotype '''
    #     return x

    def decode(self, x):
        ''' genotype -> phenotype '''
        if self.bounds is None:
            return x
        # convert [0, 1] -> [low, up]
        # try:
        low, up = self.bounds
            # return np.array([(u - l) * x_ + l for x_, l, u in zip(x, low, up)])
        return (up - low) * x + low
        # except TypeError:
        #     return np.array([(up - low) * x_ + low for x_ in x])

    @classmethod
    def set_bounds(cls, low, up):
        cls.bounds = np.array(low), np.array(up)

    @classmethod
    def set_weight(cls, weight):
        cls.weight = np.array(weight)

    @classmethod
    def clear(cls):
        cls.pool = []


@total_ordering
class Fitness(object):
    ''' 適応度
    '''
    # current_id = 0
    def __init__(self, individual):
        self.data = individual # GA個体
        self.value = None      # 適応度 (デフォルトではシーケンス型, 先頭から優先的に参照される)
        # self.id = type(self).current_id
        # type(self).current_id += 1

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return f'fitness{self.data.id}'

    def __eq__(self, other):
        if not isinstance(other, Fitness):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, Fitness):
            return NotImplemented
        return self.value < other.value

    def dominates(self, other):
        ''' 個体同士の優越関係を調べる
        関係性は次のいずれか
            0. 評価値が同一
            1. self が優越する
            2. other が優越する
            3. 互いに優越しない
        1. の場合はTrue, それ以外はFalseを返す
        '''
        return self.data <= other.data and self.data != other.data
        # return self.data < other.data

    def set_fitness(self, fitness, rank):
        self.value = fitness
        self.rank = rank

    def get_indiv(self):
        return self.data

    # def get_gene(self):
    #     ''' getter of genotype '''
    #     return self.gene

    # def get_variable(self):
    #     ''' getter of phenotype '''
    #     return self.decode(self.gene)

    # def encode(self, x):
    #     ''' phenotype -> genotype '''
    #     return x

    # def decode(self, x):
    #     ''' genotype -> phenotype '''
    #     return x

    # def evaluate(self, problem):
    #     if not self.value:
    #         self.problem = problem
    #         self.value = np.array(problem(self.get_variable()))


class Couple(object):
    def __init__(self, data):
        self._data = data
        self._offspring = []


class NoisingIndividual(Individual):
    def evaluate(self, function):
        res = super().evaluate(function)
        self.wvalue_stored = self.wvalue
        self.add_noise()
        return res

    def add_noise(self):
        self.wvalue = self.wvalue_stored + np.random.uniform(-1e-5, 1e-5,
                                                             size=len(self))

    # def __le__(self, other):
    #     self.wvalue = self.wvalue_stored + np.random.uniform(-1e-5, 1e-5,
    #                                                          size=len(self))
    #     return super().__le__(other)


class NoisingFitness(Fitness):
    def __init__(self, src):
        self.__dict__.update(src.__dict__)

    def dominates(self, other):
        self.data.add_noise()
        return super().dominates(other)
