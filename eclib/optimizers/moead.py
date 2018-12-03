'''
設計変数: 任意
目的関数: 1次元配列
解集団形状: 1次元配列, 2重配列(島モデル)

各個体にランク(自然数), 混雑度(実数)が割り当てられる
集団評価値(参考): ハイパーボリューム

解集団 =~ [個体]

選択
[個体] => [個体]

個体が生成されるのは次の3通りのいずれか
初期化: [] => 個体
交叉: [個体] => 個体 / [個体] => [個体]
突然変異: 個体 => 個体
'''

import argparse
import os
import pickle
import random
import shutil
from itertools import islice
from operator import attrgetter, itemgetter

import numpy as np

from ..base import Individual
from ..base import Population
from ..base import NondominatedSortIterator
from ..base import CrowdingDistanceCalculator
# from ..operations import SelectionIterator
# from ..operations import MatingIterator
from .iterators import SelectionIterator
from .iterators import MatingIterator


# デフォルト用
from eclib.operations import UniformInitializer
from eclib.operations import RandomSelection
from eclib.operations import RouletteSelection
from eclib.operations import TournamentSelection
from eclib.operations import TournamentSelectionStrict
from eclib.operations import TournamentSelectionDCD
from eclib.operations import BlendCrossover
from eclib.operations import SimulatedBinaryCrossover
from eclib.operations import PolynomialMutation

# default_selection = TournamentSelection(ksize=2)
default_selection = TournamentSelectionStrict(ksize=2)
# default_selection = TournamentSelectionDCD()
# default_crossover = BlendCrossover(alpha=0.5)
default_crossover = SimulatedBinaryCrossover(rate=0.9, eta=20)
default_mutation = PolynomialMutation(rate=0.05, eta=20)


def clip(x):
    return np.clip(x, 0.0, 1.0)


################################################################################


def scalar_weighted_sum(indiv, weight, ref_point):
    return -np.sum(weight * np.abs(indiv.wvalue - ref_point))

def scalar_chebyshev(indiv, weight, ref_point):
    return -np.max(weight * np.abs(indiv.wvalue - ref_point))

def scalar_boundaryintersection(indiv, weight, ref_point):
    ''' norm(weight) == 1
    '''
    nweight = weight / np.linalg.norm(weight)

    bi_theta = 5.0
    d1 = np.abs(np.dot((indiv.wvalue - ref_point), nweight))
    d2 = np.linalg.norm(indiv.wvalue - (ref_point - d1 * nweight))
    return -(d1 + bi_theta * d2)


################################################################################

class MOEAD(object):
    ''' MOEADモデル(2D)
    '''
    def __init__(self, popsize=None, problem=None, pool=None, ksize=3,
                 scalar=scalar_chebyshev,
                 selection=default_selection,
                 crossover=default_crossover,
                 mutation=default_mutation):
        self.popsize = popsize
        self.ksize = ksize
        self.problem = problem

        self.nobj = 2
        # self.select = selection
        # self.mate = crossover
        # self.mutate = mutation
        self.scalar = scalar

        self.n_parents = 2       # 1回の交叉の親個体の数
        self.n_cycle = 2         # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'new' # 世代交代方法

        # self.indiv_type = indiv_type # 個体の型
        # self.pop_type = Population # 解集団の型

        # self.initializer = None
        # self.population = Population(capacity=popsize)
        # self.next_population = Population(capacity=popsize)

        self.select_it = SelectionIterator(selection=selection, pool=pool)
        self.mate_it = MatingIterator(crossover=crossover,
                                      mutation=mutation,
                                      pool=pool)
        self.sort_it = NondominatedSortIterator
        self.share_fn = CrowdingDistanceCalculator(key=attrgetter('data')) # Fitness -> Individual

        # self.generation = 0
        # self.history = []
        if not self.popsize:
            self.init_weight2d()

    def __call__(self, population):
        if not self.popsize:
            self.popsize = len(population)
            self.init_weight2d()

        next_population = self.advance(population)
        return next_population

    # def __getitem__(self, key):
    #     return self.history[key]

    # def __len__(self):
    #     return len(self.history)

    # def setup(self, problem):
    #     ''' 最適化問題を登録 '''
    #     self.problem = problem

    def init_weight2d(self, popsize=None, ksize=None):
        ''' 重みベクトルと近傍テーブルの初期化
        '''
        if popsize:
            self.popsize = popsize
        if ksize:
            self.ksize = ksize
        if not self.popsize or not self.ksize:
            return

        def get_neighbor(index):
            imin = min(max(index-(self.ksize-1)//2, 0),
                       self.popsize-self.ksize)
            return list(range(imin, imin+self.ksize))

        self.weight = np.array([[i+1, self.popsize-i]
                               for i in range(self.popsize)])
        self.table = np.array([get_neighbor(i) for i in range(self.popsize)])
        self.ref_point = np.full(self.nobj, 'inf', dtype=np.float64)

        # self.weight = self.weight / np.linalg.norm(self.weight, axis=1)
        # print(self.ref_point)
        # exit()

    def update_reference(self, indiv):
        try:
            self.ref_point = np.min([self.ref_point, np.array(indiv.wvalue)],
                                    axis=0)
        except:
            print(self.ref_point.dtype)
            print(self.ref_point)
            print(np.array(indiv.wvalue).dtype)
            print(np.array(indiv.wvalue))
            print([self.ref_point, np.array(indiv.wvalue)])
            raise
            exit()

    def init_population(self, creator, popsize=None):
        ''' 初期集団生成
        '''
        # self.generation = 1

        # if initializer:
        #     self.initializer = initializer
        # if not self.initializer:
        #     raise Exception('initializer in None')

        if popsize:
            self.popsize = popsize
            self.init_weight2d()

        population = Population(capacity=self.popsize)

        while not population.filled():
            indiv = creator()
            fitness = indiv.evaluate(self.problem)
            population.append(fitness)

        # self.calc_fitness(population)
        self.ref_point = np.min([fit.data.wvalue for fit in population], axis=0)
        return population

    def advance(self, population):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代
        '''
        # self.generation += 1

        next_population = Population(capacity=self.popsize)
        # select_it = self.select_it(self.population, reset_cycle=self.n_cycle)
        # select_it = iter(select_it) # Fixed

        for i in range(self.popsize):
            # subpopulation = [self.population[j] for j in self.table[i]]
            child_fit = self.get_offspring(i, population, self.weight[i])
            next_population.append(child_fit)

        return next_population

    # def alternate(self, next_population):
    #     raise
    #     ''' 適応度計算 → 世代交代
    #     1. 親世代を子世代で置き換える
    #     2. 親世代と子世代の和からランクを求める
    #     '''
    #     if self.alternation == 'replace':
    #         self.calc_fitness(next_population)
    #         return next_population

    #     elif self.alternation == 'join':
    #         joined = self.population + next_population
    #         next_population = self.calc_fitness(joined, n=self.popsize)
    #         # print([fit.data.id for fit in next_population])
    #         # exit()
    #         return Population(next_population, capacity=self.popsize)

    #     else:
    #         print('Unexpected alternation type:', self.alternation)
    #         raise Exception('UnexpectedAlternation')

    def get_offspring(self, index, population, weight):
        ''' 各個体の集団内における適応度を計算する
        1. スカラー化関数
        交叉，突然変異を行い最良個体を1つ返す
        * 2018.11.21 新しい解が古い解より良い場合に置き換える処理に変更
        '''
        subpopulation = [population[i] for i in self.table[index]]
        # weight = self.weight[index]

        for i, fit in enumerate(subpopulation):
            fit_value = self.scalar(fit.data, weight, self.ref_point)
            fit.set_fitness((fit_value,), 1)

        select_it = self.select_it(subpopulation, reset_cycle=self.n_cycle)
        paretns = list(islice(select_it, self.n_parents))

        # offspring = []

        child = random.choice(list(self.mate_it(paretns)))
        # for child in self.mate_it(paretns):
        child_fit = child.evaluate(self.problem)
        self.update_reference(child)
        fit_value = self.scalar(child_fit.data, weight, self.ref_point)
        child_fit.set_fitness((fit_value,), 1)
        # offspring.append(child_fit)

        if self.alternation == 'new':
            return max(population[index], child_fit)
        elif self.alternation == 'all':
            return max(*subpopulation, child_fit)
        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')


    def get_individuals(self):
        ''' 現在の解集団(Fitness)から個体集団(Individual)を取得
        '''
        return [fit.get_indiv() for fit in self.population]

    def get_elite(self):
        return [x for x in self.population if x.rank == 1]

    def calc_rank(self, population, n=None):
        ''' 各個体の集団内におけるランクを計算して設定する
        外部から呼ぶ
        '''
        for i, front in enumerate(self.sort_it(population)):
            rank = i + 1
            for fit in front:
                fit.rank = rank
        return population

    def clear(self):
        self.Population = Population(capacity=self.popsize)
        self.history = []
        self.indiv_type.clear()

    def save(self, file):
        dirname = os.path.dirname(file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        try:
            with open(file, 'rb') as f:
                return pickle.load(f)

        except FileNotFoundError:
            print('MOEAD.load: File is not found')
            raise


################################################################################

# def hypervolume


################################################################################

def __test__():
    initializer = Initializer(10)
    print(initializer())


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', default='new_script',
                        help='Filename of the new script')
    parser.add_argument('--force', '-f', action='store_true',
                        help='force')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    '''
    docstring for main.
    '''
    args = get_args()

    if args.test:
        __test__()
        return

    file = args.out

    if not os.path.splitext(file)[1] == '.py':
        file = file + '.py'

    if args.force:
        pass
        # if os.path.exists(file):

    shutil.copy(__file__, file)
    print('create:', file)


if __name__ == '__main__':
    main()
