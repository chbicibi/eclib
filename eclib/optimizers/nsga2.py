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
from itertools import chain
from operator import attrgetter, itemgetter

import numpy as np

from ..base import Individual
from ..base import Population
from ..base import NondominatedSort
from ..base import CrowdingDistanceCalculator


def clip(x):
    return np.clip(x, 0.0, 1.0)


################################################################################

class NSGA2(object):
    ''' NSGA-IIモデル '''

    def __init__(self, popsize, selection, crossover, mutation,
                 indiv_type=Individual):
        self.popsize = popsize
        self.select = selection
        self.mate = crossover
        self.mutate = mutation

        self.n_parents = 2        # 1回の交叉の親個体の数
        self.n_cycle = 2          # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'join' # 世代交代方法

        self.indiv_type = indiv_type # 個体の型
        # self.pop_type = Population # 解集団の型

        # self.initializer = None
        self.population = Population(capacity=popsize)
        # self.next_population = Population(capacity=popsize)

        self.sort_fn = NondominatedSort(popsize)
        self.share_fn = CrowdingDistanceCalculator(key=attrgetter('data')) # Fitness -> Individual
        self.history = []

    def __getitem__(self, key):
        return self.history[key]

    def __len__(self):
        return len(self.history)

    def setup(self, problem):
        ''' 最適化問題を登録 '''
        self.problem = problem

    def create_initial_population(self, initializer=None):
        ''' 初期集団生成 '''
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
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代 '''

        next_population = Population(capacity=self.popsize)
        parents_it = self.parents_it(self.population, cycle=self.n_cycle)

        while not next_population.filled():
            # print('#1.1')
            parents = [next(parents_it) for _ in range(self.n_parents)]
            # print('#1.2')
            genomes = self.mate([fit.data for fit in parents]) # Indivに変換
            # print('#1.3')
            for genome in genomes:
                gen = self.mutate(genome)
                # print('#1.4')
                gen = np.clip(gen, 0.0, 1.0)
                # parents -> self.mate -> self.mutate

                origin = parents, self.mate, self.mutate
                offspring = self.indiv_type(gen, origin=origin)

                # print('#1.5')
                fitness = offspring.evaluate(self.problem)
                next_population.append(fitness)

        self.population = self.alternate(next_population)
        self.history.append(self.population)

    def parents_it(self, population, cycle=None):
        '''
        inputs:
            population: sequence of Fitness
            cycle: period of reset
        '''
        rest = []
        i = 0
        while True:
            if not rest or (cycle and i == cycle):
                rest = list(population)
                i = 0
            parent, rest = self.select(rest)
            i += 1
            yield parent

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
            return Population(next_population, capacity=self.popsize)

        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')

    def calc_fitness_0(self, population, n=None):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        fronts = self.sort_fn(population, n=n)
        lim = len(population) if n is None else n
        selected = []
        for i, front in enumerate(fronts):
            rank = i + 1
            fit = 0.8 ** i # TODO: 可変にする
            if self.share_fn:
                crowdings = self.share_fn(front)
            else:
                crowdings = None
            for j, indiv in enumerate(front):
                if crowdings is not None:
                    crowding = crowdings[j]
                    fitness = fit, crowding
                else:
                    fitness = fit,
                indiv.set_fitness(fitness, rank)
            lim -= len(front)
            if lim < 0:
                selected.extend(sorted(front, key=itemgetter(1),
                                       reverse=True)[:lim])
                break
            else:
                selected.extend(front)
                if lim == 0:
                    break
        return selected

    def calc_fitness(self, population, n=None):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        fronts = self.sort_fn(population, n=n)
        lim = len(population) if n is None else n
        for i, front in enumerate(fronts):
            rank = i + 1
            fit = 0.8 ** i # TODO: 可変にする
            if self.share_fn:
                crowdings = self.share_fn(front)
            else:
                crowdings = None
            for j, indiv in enumerate(front):
                if crowdings is not None:
                    crowding = crowdings[j]
                    fitness = fit, crowding
                else:
                    fitness = fit,
                indiv.set_fitness(fitness, rank)

        chosen = list(chain(*fronts[:-1]))
        k = lim - len(chosen)
        if k > 0:
            sorted_front = sorted(fronts[-1], key=itemgetter(1), reverse=True)
            chosen.extend(sorted_front[:k])
        elif k < 0:
            print(k)
            exit()
        return chosen
    # def get_parents(self, population, n):
    #     '''
    #     inputs:
    #         population: sequence of Fitness
    #     result:
    #         selected: sequence of Fitness
    #     '''
    #     rest = list(population)
    #     selected = []
    #     for i in range(n):
    #         parent, rest = self.select(rest)
    #         if not parent:
    #             return selected
    #         selected.append(parent)
    #     return selected

    def get_individuals(self):
        return [fit.data for fit in self.population]

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

    def load(file):
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
