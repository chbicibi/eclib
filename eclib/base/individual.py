from functools import total_ordering
# import numpy as np
from .collections import Container


class Genome(list):
    ''' GA遺伝子を格納するデータ構造
    '''
    def __init__(self):
        pass


class Individual(Container):
    ''' 進化計算個体
    遺伝子と評価値を管理する
    '''
    current_id = 0
    pool = []

    def __init__(self, genome, origin=None):
        super().__init__()
        self.genome = genome # 遺伝子型
        self.origin = origin # 派生元 (初期化関数又は関数と引数の組)
        self.value = None    # 評価値 (デフォルトではシーケンス型)

        # self.id = Individual.current_id
        # Individual.current_id += 1
        self.id = len(Individual.pool)
        Individual.current_id = self.id + 1
        Individual.pool.append(self)

    def __str__(self):
        return f'indiv{self.id}'

    # 以下の6つの比較用メソッドは優越関係を表すものでそれぞれ対になるメソッドの否定ではない
    # self < other != not self <= other
    def __eq__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s == o for s, o in zip(self.value, other.value))

    def __ne__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return any(s != o for s, o in zip(self.value, other.value))

    def __lt__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s < o for s, o in zip(self.value, other.value))

    def __le__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s <= o for s, o in zip(self.value, other.value))

    def __gt__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s > o for s, o in zip(self.value, other.value))

    def __ge__(self, other):
        if not isinstance(other, Individual):
            # return NotImplemented
            raise TypeError
        return all(s >= o for s, o in zip(self.value, other.value))

    def evaluate(self, function):
        if not self.evaluated():
            self.function = function
            self.value = function(self.get_variable())
        return Fitness(self)

    def get_variable(self):
        ''' getter of phenotype '''
        return self.decode(self.genome)

    def evaluated(self):
        return self.value is not None

    def get_gene(self, **kwargs):
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
        return x


@total_ordering
class Fitness(Container):
    ''' 適応度
    '''
    def __init__(self, individual):
        super().__init__()
        self.data = individual # GA個体
        self.value = None      # 適応度 (デフォルトではシーケンス型, 先頭から優先的に参照される)

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

    # def initialize(self, initializer):
    #     self.gene = initializer()

    def set_fitness(self, fitness, rank):
        self.value = fitness
        self.rank = rank

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