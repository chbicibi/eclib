#! /usr/bin/env python3

'''
'''

import argparse
import glob
import os
import shutil
import sys
from itertools import chain
from operator import attrgetter

import numpy as np
import matplotlib.pyplot as plt

from eclib.benchmarks import rosenbrock, zdt1, zdt2, zdt3, zdt4, zdt6
from eclib.operations import UniformInitializer
# from eclib.operations import RouletteSelection
# from eclib.operations import TournamentSelection
# from eclib.operations import TournamentSelectionStrict
# from eclib.operations import TournamentSelectionDCD
# from eclib.operations import BlendCrossover
# from eclib.operations import SimulatedBinaryCrossover
# from eclib.operations import PolynomialMutation
from eclib.base import Individual
from eclib.base import Fitness
from eclib.base import Environment
from eclib.base import Creator
from eclib.base import Population
from eclib.optimizers import NSGA2
from eclib.optimizers import MOEAD

import myutils as ut


################################################################################

def main1(out='result', force=False, clear_directory=False):
    ''' GAテスト & プロット
    '''
    ###
    def ln(ind, origin):
        if not origin:
            return []
        return list(zip(origin[0].value, ind.value, origin[1].value))
    def dist(ind, origin):
        if not origin:
            return 0
        values = np.array([[ind.value, par.value] for par in origin])
        diff = values[:, :, 0] - values[:, :, 1]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return np.sum(dists)

    fig, ax = plt.subplots()
    def plot(pop):
        pairs = [(fit.data, fit.data.origin.origin or ()) for fit in pop]
        parents = list(chain(*(x[1] for x in pairs)))
        lines = [ln(ind, origin) for ind, origin in pairs]
        # print(lines)
        # exit()
        if lines and [x for x in lines if x]:
            dists = [dist(ind, origin) for ind, origin in pairs]
            print(sum(dists))

        ax.cla()
        cm = plt.get_cmap('jet')
        # print(cm(10))
        # exit()

        if parents:
            x_p, y_p = np.array([ind.value for ind in parents]).T
            ax.scatter(x_p, y_p, c='r')

            for i, l in enumerate(lines):
                if l:
                    plt.plot(*l, c=cm(i/(len(lines)+1)), linewidth=0.5)

        x, y = np.array([fit.data.value for fit in pop]).T
        ax.scatter(x, y, c='b')
        plt.pause(1e-10)

    ###
    def local_main():
        n_dim = 30
        popsize = 100
        problem = zdt1

        with Environment() as env:
            # 個体クラス
            indiv_type = Individual
            # 初期個体生成クラス
            indiv_pool = env.register(indiv_type)

            # 遺伝子生成クラス
            initializer = UniformInitializer(n_dim)

            creator = Creator(initializer, indiv_pool)

            # # 適応度クラス
            # fit_type = Fitness
            # # 初期個体生成クラス
            # evaluator = env.register(fit_type)

            ###

            # optimizer = NSGA2(problem=problem, pool=indiv_pool)
            optimizer = MOEAD(problem=problem, pool=indiv_pool, ksize=5)
            # optimizer.set_initializer(Initializer(3))
            # optimizer.setup(problem)

            ### Additional setting ###
            # optimizer.initializer = initializer
            # optimizer.n_cycle = None
            # optimizer.alternation = 'replace'
            ##########################

            # indivs = [creator() for _ in range(popsize)]
            # population = Population([ind.evaluate(problem) for ind in indivs])
            # optimizer.calc_fitness(population)
            population = optimizer.init_population(creator, popsize=popsize)
            history = [population]

            # print(population[0].data.origin)
            # return

            for i in range(100):
                population = optimizer(population)
                plot(population)
                origin = population[0].data.origin.origin or []
                # print([x.id for x in origin], '->', population[0].data.id)
                history.append(population)

                if i % 50 == 50-1:
                    ut.save(f'temp{i}.pkl', history)

            plt.show()
            return env, optimizer, history
    ###

    def resume_main(env, optimizer, history):
        print('resume_main')
        for i, population in enumerate(history):
            plot(population)
            origin = population[0].data.origin.origin or []
            # print([x.id for x in origin], '->', population[0].data.id)
        plt.show()

    file = 'test_moead.pkl'
    if os.path.exists(file) and not force:
        env, optimizer, history = ut.load(file)
        resume_main(env, optimizer, history)

    else:
        env, optimizer, history = local_main()
        ut.save(file, (env, optimizer, history))


################################################################################

def __test__():
    a = np.array([[[1, 2], [10, 20]],
                  [[3, 4], [30, 40]],
                  [[5, 6], [50, 60]]])
    print(np.sum(a, axis=0))


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='?', default='',
                        help='Main method type')
    parser.add_argument('--out', '-o', default='result',
                        help='Filename of the new script')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove output directory before start')
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

    # print(sys.getrecursionlimit())
    sys.setrecursionlimit(80)

    args = get_args()
    out = args.out
    clear = args.clear

    if args.test:
        __test__()
        return

    if args.method == '1':
        main1(out=out, force=args.force)


if __name__ == '__main__':
    main()
