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

class NSGA2(object):
    ''' NSGA-IIモデル '''

    def __init__(self, popsize=None, problem=None, pool=None,
                 selection=default_selection,
                 crossover=default_crossover,
                 mutation=default_mutation):
        self.popsize = popsize
        self.problem = problem
        # self.selection = selection
        # self.crossover = crossover
        # self.mutation = mutation

        self.n_parents = 2        # 1回の交叉の親個体の数
        self.n_cycle = 2          # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'join' # 世代交代方法

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

    def __call__(self, population):
        if not self.popsize:
            self.popsize = len(population)

        next_population = self.advance(population)
        return self.alternate(population, next_population)

    # def __getitem__(self, key):
        # return self.history[key]

    # def __len__(self):
    #     return len(self.history)

    # def setup(self, problem):
    #     ''' 最適化問題を登録 '''
    #     self.problem = problem

    def init_population(self, creator, popsize=None):
        ''' 初期集団生成
        '''
        if popsize:
            self.popsize = popsize
        # self.generation = 1

        population = Population(capacity=self.popsize)

        # if initializer:
        #     self.initializer = initializer
        # if not self.initializer:
        #     raise Exception('initializer in None')

        while not population.filled():
            indiv = creator()
            fitness = indiv.evaluate(self.problem)
            population.append(fitness)

        self.calc_fitness(population)
        return population

    def advance(self, population):
        ''' 選択→交叉→突然変異→評価(→適応度計算→世代交代)
        '''
        # self.generation += 1

        next_population = Population(capacity=self.popsize)
        select_it = self.select_it(population, reset_cycle=self.n_cycle)
        select_it = iter(select_it) # Fixed

        while not next_population.filled():
            parents_it = list(islice(select_it, self.n_parents)) # Fixed

            for child in self.mate_it(parents_it):
                child_fit = child.evaluate(self.problem)
                next_population.append(child_fit)

        return next_population

    def alternate(self, population, next_population):
        ''' 適応度計算 → 世代交代
        1. 親世代を子世代で置き換える
        2. 親世代と子世代の和からランクを求める
        '''
        if self.alternation == 'replace':
            self.calc_fitness(next_population)
            return next_population

        elif self.alternation == 'join':
            joined = population + next_population
            next_population = self.calc_fitness(joined, n=self.popsize)
            # print([fit.data.id for fit in next_population])
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
            fit_value = -i # TODO: 可変にする
            # if i == 0:
            #     print('len(i==0):', len(front), ' ')

            if self.share_fn:
                it = self.share_fn(front)
                try:
                    for fit, crowding in zip(front, it):
                        fitness = fit_value, crowding
                        # print(fitness)
                        fit.set_fitness(fitness, rank)
                except:
                    print('Error')
                    print(front)
                    print(it)
                    raise
            else:
                for fit in front:
                    fitness = fit_value,
                    fit.set_fitness(fitness, rank)

            lim -= len(front) # 個体追加後の余裕
            if lim >= 0:
                selected.extend(front)
                if lim == 0:
                    return selected
            # elif i == 0:
            #     return front
            else:
                front.sort(key=itemgetter(1), reverse=True) # 混雑度降順で並べ替え
                # print([itemgetter(1)(fit) for fit in front])
                # exit()
                selected.extend(front[:lim])
                return selected

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
