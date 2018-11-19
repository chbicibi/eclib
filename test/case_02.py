#! /usr/bin/env python3

'''
'''

import argparse
import glob
import os
import shutil
import sys
from operator import attrgetter

import numpy as np
import matplotlib.pyplot as plt

from eclib.benchmarks import rosenbrock, zdt1, zdt2, zdt3, zdt4, zdt6
# from eclib.operations import UniformInitializer
# from eclib.operations import RouletteSelection
# from eclib.operations import TournamentSelection
# from eclib.operations import TournamentSelectionStrict
# from eclib.operations import TournamentSelectionDCD
# from eclib.operations import BlendCrossover
# from eclib.operations import SimulatedBinaryCrossover
# from eclib.operations import PolynomialMutation
from eclib.optimizers import MOEAD
# from eclib.base import Individual

import base_01 as base
import myutils as ut


################################################################################

def ga_main01(out='result', clear_directory=False):
    ''' GAテスト & プロット
    '''
    if clear_directory and os.path.isdir(out):
        shutil.rmtree(out)

    problem = zdt6
    epoch = 250
    save_trigger = lambda i: i == epoch # 最後だけ

    with base.MOEAD_ENV(problem) as optimizer:
        with ut.stopwatch('main'):
            # GA開始
            # 初期集団生成
            optimizer.init_population()

            # 進化
            for i in range(1, epoch + 1):
                population = optimizer.advance()
                print('epoch:', i, 'popsize:', len(population), end='\r')
                # print(len(set([x.id for x in optimizer.get_individuals()])))
                if save_trigger(i):
                    optimizer.save(file=os.path.join(out, f'epoch{i}.pickle'))


        # elite = optimizer.get_elite()
        # history = optimizer.history
        # def best(pop):
        #     return [x for x in pop if x.rank == 1][0]()
        # bests = np.array([best(pop) for pop in history])

        # first_population = optimizer[0]

        last_population = optimizer.get_individuals()
        last_population.sort(key=lambda x: x.value)
        optimal_front = base.get_optomal_front(f'pareto_front/{problem.__name__}_front.json')

        print(len(set([ind.id for ind in last_population])))

        ### TEMP: check stat ###
        print("Convergence: ", base.convergence(last_population, optimal_front))
        print("Diversity: ", base.diversity(last_population, optimal_front[0],
                                            optimal_front[-1]))
        ########################

        ### TEMP: plot front ###
        x, y = np.array([x.value for x in last_population]).T

        plt.scatter(optimal_front[:, 0], optimal_front[:, 1], c='r')

        plt.scatter(x, y, c='b')
        plt.axis("tight")
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        plt.show()
        ########################


def ga_main02(out='result', clear_directory=False):
    ''' GAテスト & プロット
    50回の平均
    '''
    if clear_directory and os.path.isdir(out):
        shutil.rmtree(out)

    problem = zdt1
    epoch = 250
    save_trigger = lambda i: i == epoch # 最後だけ
    optimal_front = base.get_optomal_front('pareto_front/zdt1_front.json')
    stat = []

    for rep in range(50):
        with base.MOEAD_ENV(problem) as optimizer:
            with ut.stopwatch(f'epoch{epoch+1}'):
                optimizer.init_population()
                for i in range(1, epoch + 1):
                    population = optimizer.advance()
                    print('epoch:', i, 'popsize:', len(population),
                          end='\r')
            optimizer.save(file=os.path.join(out, f'main02_rep{rep}.pkl'))

            last_population = optimizer.get_individuals()
            last_population.sort(key=lambda x: x.value)

            conv = base.convergence(last_population, optimal_front)
            div = base.diversity(last_population, optimal_front[0],
                                 optimal_front[-1])
            stat.append((conv, div))

            print("Convergence: ", conv)
            print("Diversity: ", div)

    print('=' * 20, 'Average', '=' * 20)
    print("Convergence: ", np.mean([x[0] for x in stat]))
    print("Diversity: ",  np.mean([x[1] for x in stat]))


def ga_main1(out='result'):
    file = ut.fsort(glob.glob(os.path.join(out, f'epoch*.pickle')))[-1]
    optimizer = MOEAD.load(file=file)

    elite = optimizer.get_elite()
    print('elite:', len(elite))

    for epoch, population in enumerate(optimizer):
        print(epoch, end='\n')
        plt.cla()

        # print([fit.rank for fit in population])
        # exit()

        # for i in range(1):
        #     front = [x for x in population if x.rank == i]
        #     if not front:
        #         continue
        #     x, y = np.array([x.data.value for x in front]).T
        #     plt.scatter(x, y, label=f'{front[0][0]:.3f}')
        front = population
        x, y = np.array([x.data.value for x in front]).T
        plt.scatter(x, y, label=f'{front[0][0]:.3f}')

        # plt.legend()
        if epoch >= 10:
            plt.show()
            return

        if epoch < len(optimizer) - 1:
            plt.pause(0.2)
        else:
            plt.show()


def ga_main2(out='result'):
    file = ut.fsort(glob.glob(os.path.join(out, f'epoch*.pickle')))[-1]
    optimizer = MOEAD.load(file=file)

    population = optimizer[-1]
    front = [x for x in population if x.rank == 1]
    front.sort(key=attrgetter('data.value'))

    for ind in front:
        print(ind.value, ind.data.value)

    crowdings = [ind.value[1] for ind in front]

    fig, axes = plt.subplots(2)
    axes[0].plot(crowdings)

    x, y = np.array([x.data.value for x in front]).T
    im = axes[1].scatter(x, y, c=crowdings, cmap='jet')
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    fig.colorbar(im)
    plt.show()

    # for ind in first_population:
    #     print(ind(), ind.rank, ind.fitness)
    # return

    # last_population = optimizer.population
    # x, y = np.array([x() for x in last_population]).T
    # plt.scatter(x, y)

    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.show()


################################################################################

def __test__():
    from scipy.interpolate import RegularGridInterpolator
    from mpl_toolkits.mplot3d import Axes3D

    x = np.arange(0, 1, 0.2)
    y = np.arange(0, 1, 0.2)

    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock([vx, vy]) for vy in y] for vx in x])

    # print(X.shape)
    # # print(y.shape)
    # print(Z.shape)
    # return
    xx = np.arange(0, 0.8, 0.01)
    yy = np.arange(0, 0.8, 0.01)
    XX, YY = np.meshgrid(xx, yy)
    # ax.scatter3D(*map(lambda a: a.reshape(-1), (X, Y, Z)))

    zz = RegularGridInterpolator((x, y), Z)

    ZZ = np.array([[zz([vx, vy]) for vy in yy] for vx in xx])[:, :, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.plot_surface(XX, YY, ZZ)
    ax.scatter3D(*map(lambda a: a.reshape(-1), (XX, YY, ZZ)))

    ax.set_title("Scatter Plot")
    plt.show()

    # plt.scatter(z[:, 0], z[:, 1])
    # plt.xlim((0, 1))
    # # plt.ylim((0, 1))
    # plt.show()


class TestClass(object):
    def __new__(cls, *args):
        print('new', args)
        return super().__new__(cls)

    def __init__(self, *args):
        self.value0 = [1, 2, 3]
        self.value1 = [10, 20, 30]
        print('init', args)

    def __getitem__(self, key):
        return self.value0[key]

    def __iter__(self):
        return iter(self.value1)


class TestIterator(object):
    def __init__(self, *args):
        print('init', args)
        self.value0 = [10, 20, 30]

    # def __call__(self, *args):
    #     print('call', args)
    #     for i in self.value0:
    #         yield i

    def __iter__(self):
        print('iter')
        for i in self.value0:
            yield i


def __test__():
    print(np.min([[0, 15], (5, 10)], axis=0))


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
    sys.setrecursionlimit(10000)

    args = get_args()
    out = args.out
    clear = args.clear

    if args.test:
        __test__()
        return

    if args.method == '0':
        ga_main(out=out, clear_directory=clear)
    elif args.method == '01':
        ga_main01(out=out, clear_directory=clear)
    elif args.method == '02':
        ga_main02(out=out, clear_directory=clear)
    elif args.method == '1':
        ga_main1(out=out)
    elif args.method == '2':
        ga_main2(out=out)


if __name__ == '__main__':
    main()
