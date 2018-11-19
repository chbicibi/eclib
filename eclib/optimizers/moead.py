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
from .nsga2 import SelectionIterator
from .nsga2 import MatingIterator


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
    def __init__(self, popsize, selection, crossover, mutation, ksize,
                 indiv_type=Individual):
        self.popsize = popsize
        self.ksize = ksize
        self.nobj = 2
        # self.select = selection
        # self.mate = crossover
        # self.mutate = mutation
        self.scalar = scalar_boundaryintersection

        self.n_parents = 2        # 1回の交叉の親個体の数
        self.n_cycle = 2          # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'join' # 世代交代方法

        self.indiv_type = indiv_type # 個体の型
        # self.pop_type = Population # 解集団の型

        # self.initializer = None
        self.population = Population(capacity=popsize)
        # self.next_population = Population(capacity=popsize)

        self.select_it = SelectionIterator(selection=selection)
        self.mate_it = MatingIterator(crossover=crossover,
                                      mutation=mutation,
                                      indiv_type=indiv_type)
        self.sort_it = NondominatedSortIterator
        self.share_fn = CrowdingDistanceCalculator(key=attrgetter('data')) # Fitness -> Individual

        self.generation = 0
        self.history = []
        self.init_weight2d()

    def __getitem__(self, key):
        return self.history[key]

    def __len__(self):
        return len(self.history)

    def setup(self, problem):
        ''' 最適化問題を登録 '''
        self.problem = problem

    def init_weight2d(self):
        ''' 重みベクトルと近傍テーブルの初期化
        '''
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


    def init_population(self, initializer=None):
        ''' 初期集団生成
        '''
        self.generation = 1

        if initializer:
            self.initializer = initializer
        if not self.initializer:
            raise Exception('initializer in None')

        while not self.population.filled():
            indiv = self.indiv_type(self.initializer(), origin=self.initializer)
            fitness = indiv.evaluate(self.problem)
            self.population.append(fitness)

        # self.calc_fitness(self.population)
        self.ref_point = np.min([fit.data.wvalue for fit in self.population],
                                axis=0)
        self.history.append(self.population)
        # print(self.ref_point)
        # exit()

    def advance(self):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代
        '''
        self.generation += 1

        next_population = Population(capacity=self.popsize)
        # select_it = self.select_it(self.population, reset_cycle=self.n_cycle)
        # select_it = iter(select_it) # Fixed

        for i in range(self.popsize):
            subpopulation = [self.population[j] for j in self.table[i]]
            child_fit = self.get_offspring(subpopulation, self.weight[i])
            next_population.append(child_fit)

        self.population = next_population
        self.history.append(self.population)
        # exit()
        return self.population

    def alternate(self, next_population):
        raise
        ''' 適応度計算 → 世代交代
        1. 親世代を子世代で置き換える
        2. 親世代と子世代の和からランクを求める
        '''
        if self.alternation == 'replace':
            self.calc_fitness(next_population)
            return next_population

        elif self.alternation == 'join':
            joined = self.population + next_population
            next_population = self.calc_fitness(joined, n=self.popsize)
            # print([fit.data.id for fit in next_population])
            # exit()
            return Population(next_population, capacity=self.popsize)

        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')

    def get_offspring(self, subpopulation, weight):
        ''' 各個体の集団内における適応度を計算する
        1. スカラー化関数
        交叉，突然変異を行い最良個体を1つ返す
        '''
        # subpopulation = [population(i) for i in self.table[index]]
        # weight = self.weight[index]

        for i, fit in enumerate(subpopulation):
            fit_value = self.scalar(fit.data, weight, self.ref_point)
            fit.set_fitness((fit_value,), 1)

        select_it = self.select_it(subpopulation, reset_cycle=self.n_cycle)
        paretns = list(islice(select_it, self.n_parents))

        for child in self.mate_it(paretns):
            child_fit = child.evaluate(self.problem)
            self.update_reference(child)
            fit_value = self.scalar(child_fit.data, weight, self.ref_point)
            child_fit.set_fitness((fit_value,), 1)
            subpopulation.append(child_fit)

        # print([fit.value for fit in subpopulation])
        # exit()

        return max(subpopulation)


    def get_individuals(self):
        ''' 現在の解集団(Fitness)から個体集団(Individual)を取得
        '''
        return [fit.get_indiv() for fit in self.population]

    def get_elite(self):
        return [x for x in self.population if x.rank == 1]

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
            print('NSGA2.load: File is not found')
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
