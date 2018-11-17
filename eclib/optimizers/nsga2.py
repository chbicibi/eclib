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


def clip(x):
    return np.clip(x, 0.0, 1.0)


################################################################################

class PartialSelectionIterator(object):
    ''' SelectionIteratorの部分適用オブジェクト
    '''
    def __init__(self, selection):
        self._selection = selection

    def __call__(self, population, reset_cycle=None):
        return SelectionIterator(self._selection, population, reset_cycle)


class SelectionIterator(object):
    ''' 交配の親選択イテレータ
    個体集団から親個体を選択し，選択された親個体を解集団から削除する(削除方式はselection関数に依存)
    reset_cycleが与えられた場合は解をreset_cycle個生成するごとに解集団をpopulationで初期化する
    '''
    def __new__(cls, selection, population=None, reset_cycle=None):
        if population is None:
            return PartialSelectionIterator(selection)
        return super().__new__(cls)

    def __init__(self, selection, population, reset_cycle=None):
        self._selection = selection
        self._population = population
        self._reset_cycle = reset_cycle
        self._stored = []

    def __iter__(self):
        rest = []
        couner = 0
        while True:
            if not rest or (self._reset_cycle and couner == self._reset_cycle):
                rest = list(self._population)
                couner = 0
            selected, rest = self._selection(rest)
            couner += 1
            self._stored.append(selected)
            yield selected

    def __getnewargs__(self):
        return self._selection, self._population, self._reset_cycle


class PartialMatingIterator(object):
    ''' MatingIteratorの部分適用オブジェクト
    '''
    def __init__(self, crossover, mutation, indiv_type):
        self._crossover = crossover
        self._mutation = mutation
        self._indiv_type = indiv_type

    def __call__(self, origin):
        return MatingIterator(self._crossover, self._mutation, self._indiv_type,
                              origin)


class MatingIterator(object):
    ''' 親の組からこの組を生成するイテレータ
    crossoverとmutationの直列方式
    '''
    def __new__(cls, crossover, mutation, indiv_type, origin=None):
        if origin is None:
            return PartialMatingIterator(crossover, mutation, indiv_type)
            # return lambda x: cls(crossover, mutation, indiv_type, x)
        return super().__new__(cls)

    def __init__(self, crossover, mutation, indiv_type, origin):
        self._crossover = crossover
        self._mutation = mutation
        self._indiv_type = indiv_type
        self._origin = origin

        self._parents = [fit.get_indiv() for fit in origin] # Indivに変換
        self._stored = []

    def __iter__(self):
        genomes = self._crossover(self._parents) # 子個体の遺伝子の組

        for genome in genomes:
            # 1個体の遺伝子
            genome = self._mutation(genome)
            child = self._indiv_type(genome, origin=self)
            self._stored.append(child)
            yield child

    def __getnewargs__(self):
        return self._crossover, self._mutation, self._indiv_type, self._origin


################################################################################

class NSGA2(object):
    ''' NSGA-IIモデル '''

    def __init__(self, popsize, selection, crossover, mutation,
                 indiv_type=Individual):
        self.popsize = popsize
        # self.select = selection
        # self.mate = crossover
        # self.mutate = mutation

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

    def __getitem__(self, key):
        return self.history[key]

    def __len__(self):
        return len(self.history)

    def setup(self, problem):
        ''' 最適化問題を登録 '''
        self.problem = problem

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

        self.calc_fitness(self.population)
        self.history.append(self.population)

    def advance(self):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代
        '''
        self.generation += 1

        next_population = Population(capacity=self.popsize)
        select_it = self.select_it(self.population, reset_cycle=self.n_cycle)

        while not next_population.filled():
            parents_it = list(islice(select_it, self.n_parents))

            for child in self.mate_it(parents_it):
                child_fit = child.evaluate(self.problem)
                next_population.append(child_fit)

        self.population = self.alternate(next_population)
        self.history.append(self.population)
        return self.population

    def alternate(self, next_population):
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
            print([fit.rank for fit in next_population])
            # exit()
            return Population(next_population, capacity=self.popsize)

        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')

    def calc_fitness(self, population, n=None):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        lim = len(population) if n is None else n
        selected = []

        for i, front in enumerate(self.sort_it(population)):
            # print('g:', self.generation, 'i:', i, 'l:', len(front))
            rank = i + 1
            fit_value = 0.8 ** i # TODO: 可変にする

            if self.share_fn:
                for fit, crowding in zip(front, self.share_fn(front)):
                    fitness = fit_value, crowding
                    fit.set_fitness(fitness, rank)
            else:
                for fit in front:
                    fitness = fit_value,
                    fit.set_fitness(fitness, rank)

            lim -= len(front) # 個体追加後の余裕
            if lim >= 0:
                selected.extend(front)
                if lim == 0:
                    return selected
            else:
                front.sort(key=itemgetter(1), reverse=True) # 混雑度降順で並べ替え
                selected.extend(front[:lim])
                return selected

    # def calc_rank(self, population, n=None):
    #     ''' 各個体の集団内におけるランクを計算して設定する
    #     '''
    #     for i, fronts in enumerate(self.sort_it(population)):
    #         rank = i + 1
    #         for fit in fronts:
    #             fit.rank = rank
    #     return population

    def get_individuals(self):
        ''' 現在の解集団(Fitness)から個体集団(Individual)を取得
        '''
        return [fit.get_indiv() for fit in self.population]

    def get_elite(self):
        return [x for x in self.population if x.rank == 1]

    def clear(self):
        self.Population = Population(capacity=self.popsize)
        self.history = []

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
