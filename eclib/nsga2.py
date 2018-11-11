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
from operator import attrgetter

import numpy as np

from .base import Individual
from .base import Population
from .base import NondominatedSort
from .base import CrowdingDistanceCalculator


def clip(x):
    return np.clip(x, 0, 1)


################################################################################

class NSGA2(object):
    ''' NSGA-IIモデル '''

    def __init__(self, popsize, selection, crossover, mutation):
        self.popsize = popsize
        self.select = selection
        self.mate = crossover
        self.mutate = mutation

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

    # def set_initializer(self, initializer):
    #     ''' 初期集団生成用クラスを登録 '''
    #     self.initializer = initializer

    def setup(self, problem):
        ''' 最適化問題を登録 '''
        self.problem = problem

    def create_initial_population(self, initializer):
        ''' 初期集団生成 '''
        self.initializer = initializer

        while not self.population.filled():
            indiv = Individual(self.initializer(), origin=self.initializer)
            fitness = indiv.evaluate(self.problem)
            self.population.append(fitness)

        self.calc_fitness(self.population)
        self.history.append(self.population)

    def calc_fitness(self, population):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        fronts = self.sort_fn(population)
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
                    fitness = (fit, crowding)
                else:
                    fitness = fit,
                indiv.set_fitness(fitness, rank)

    def advance(self):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代 '''

        next_population = Population(capacity=self.popsize)

        while not next_population.filled():
            # print('#1.1')
            parents = self.get_parents(self.population)
            # print('#1.2')
            genomes = self.mate([fit.data for fit in parents]) # Indivに変換
            # print('#1.3')
            for genome in genomes:
                gen = self.mutate(genome)
                # print('#1.4')
                gen = np.clip(gen, 0, 1)
                # parents -> self.mate -> self.mutate
                offspring = Individual(gen,
                                       origin=(parents, self.mate, self.mutate))
                # print('#1.5')
                fitness = offspring.evaluate(self.problem)
                next_population.append(fitness)

        self.calc_fitness(next_population)

        # print('#2')

        self.population = next_population
        self.history.append(self.population)

    def advance_n(self):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代 '''

        next_population = Population(capacity=self.popsize)

        n = self.popsize//2
        # parents_n = self.get_parents_n(self.population, n)
        parents_n = []

        while not next_population.filled():
            # print('#1.1')
            if len(parents_n) < 2:
                parents_n = self.get_parents_n(self.population, n)
            parents = [parents_n.pop(0) for i in (0, 1)]

            # print('#1.2')
            genomes = self.mate([fit.data for fit in parents]) # Indivに変換
            # print('#1.3')
            for genome in genomes:
                gen = self.mutate(genome)
                # print('#1.4')
                gen = np.clip(gen, 0, 1)
                # parents -> self.mate -> self.mutate
                offspring = Individual(gen,
                                       origin=(parents, self.mate, self.mutate))
                # print('#1.5')
                fitness = offspring.evaluate(self.problem)
                next_population.append(fitness)

        self.calc_fitness(next_population)

        # print('#2')

        self.population = next_population
        self.history.append(self.population)

    def advance_e(self):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代 '''

        temp_population = Population(capacity=self.popsize)
        next_population = Population(capacity=self.popsize)

        n = self.popsize//2
        # parents_n = self.get_parents_n(self.population, n)
        parents_n = []

        while not next_population.filled():
            # print('#1.1')
            if len(parents_n) < 2:
                parents_n = self.get_parents_n(self.population, n)
            parents = [parents_n.pop(0) for i in (0, 1)]

            # print('#1.2')
            genomes = self.mate([fit.data for fit in parents]) # Indivに変換
            # print('#1.3')
            for genome in genomes:
                gen = self.mutate(genome)
                # print('#1.4')
                gen = np.clip(gen, 0, 1)
                # parents -> self.mate -> self.mutate
                offspring = Individual(gen,
                                       origin=(parents, self.mate, self.mutate))
                # print('#1.5')
                fitness = offspring.evaluate(self.problem)
                next_population.append(fitness)

        self.calc_fitness(next_population)

        while not temp_population.filled():
            selected = self.get_parents(self.population+next_population, 1)[0]
            temp_population.append(selected)

        # print('#2')

        self.population = temp_population
        self.history.append(self.population)


    def get_parents(self, population, n=2):
        '''
        inputs:
            population: sequence of Fitness
        result:
            res: pair of Fitness
        '''
        rest = list(population)
        res = []
        for i in range(n):
            parent, rest = self.select(rest)
            res.append(parent)
        return res

    def get_parents_n(self, population, n):
        '''
        inputs:
            population: sequence of Fitness
        result:
            res: pair of Fitness
        '''
        rest = list(population)
        res = []
        for i in range(n):
            parent, rest = self.select(rest)
            if not parent:
                return res
            res.append(parent)
        return res

    def get_individuals(self):
        return [fit.data for fit in self.population]

    def get_elite(self):
        return [x for x in self.population if x.rank == 1]

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
