#! /usr/bin/env python3

'''
1. 初期集団生成
2.
'''

import argparse
import glob
import os
import shutil
import sys
from operator import attrgetter

import numpy as np
import matplotlib.pyplot as plt

from eclib.benchmarks import zdt1, rosenbrock
from eclib.nsga2 import NSGA2
from eclib.operations import UniformInitializer
from eclib.operations import RouletteSelection
from eclib.operations import TournamentSelection
from eclib.operations import TournamentSelectionStrict
from eclib.operations import TournamentSelectionDCD
from eclib.operations import BLXCrossover
from eclib.operations import SimulatedBinaryBounded
from eclib.operations import PolynomialBounded

import myutils as ut

# class Individual(object):
#     ''' 進化計算個体
#     saku
#     '''

#     def __init__(self):
#         self.genotype

#     def __call__(self):
#         return np.random.rand(self._size)


# class Population(object):
#     ''' 進化計算集団 '''

#     def __init__(self, n):
#         self._size = n

#     def __call__(self):
#         return np.random.rand(self._size)


################################################################################

def identity(x):
    return x


def clip(x):
    return np.clip(x, 0, 1)


################################################################################

def ga_main(out='result', clear_directory=False):
    if clear_directory and os.path.isdir(out):
        shutil.rmtree(out)

    # パラメータ
    n_dim = 30
    pop_size = 100
    epoch = 500
    # save_trigger = lambda i: i == 1 or i % 10 == 0 # 10 epochごと
    save_trigger = lambda i: i == epoch              # 最後だけ

    # 問題
    problem = zdt1
    initializer = UniformInitializer(n_dim)
    selection = TournamentSelection(ksize=2)
    # selection = TournamentSelectionDCD()
    # crossover = BLXCrossover(alpha=0.5)
    crossover = SimulatedBinaryBounded(rate=0.9, eta=20)
    mutation = PolynomialBounded(rate=1/n_dim, eta=20)

    optimizer = NSGA2(pop_size, selection, crossover, mutation)
    # optimizer.set_initializer(Initializer(3))
    optimizer.setup(problem)

    with ut.stopwatch('main'):
        # GA開始
        # 初期集団生成
        optimizer.create_initial_population(initializer=initializer)

        # 進化
        for i in range(1, epoch + 1):
            # optimizer.advance()
            optimizer.advance()
            print('epoch:', i, end='\r')
            if save_trigger(i):
                optimizer.save(file=os.path.join(out, f'epoch{i}.pickle'))


    # elite = optimizer.get_elite()
    # history = optimizer.history
    # def best(pop):
    #     return [x for x in pop if x.rank == 1][0]()
    # bests = np.array([best(pop) for pop in history])

    # first_population = optimizer[0]

    last_population = optimizer.get_individuals()
    x, y = np.array([x.value for x in last_population]).T
    plt.scatter(x, y)
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    plt.show()


def ga_main1(out='result'):
    file = ut.fsort(glob.glob(os.path.join(out, f'epoch*.pickle')))[-1]
    optimizer = NSGA2.load(file=file)

    elite = optimizer.get_elite()
    print('elite:', len(elite))

    for epoch in range(250):
        print(epoch, end='\n')
        plt.cla()
        first_population = optimizer[epoch]

        for i in range(10):
            front = [x for x in first_population if x.rank == i]
            if not front:
                continue
            x, y = np.array([x.data.value for x in front]).T
            plt.scatter(x, y, label=f'{front[0][0]:.3f}')

        plt.legend()
        if epoch < 249:
            plt.pause(0.2)
        else:
            plt.show()


def ga_main2(out='result'):
    file = ut.fsort(glob.glob(os.path.join(out, f'epoch*.pickle')))[-1]
    optimizer = NSGA2.load(file=file)

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


# def __test__():
#     a = list(range(5))
#     print(a.pop(3))
#     print(a)


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
    elif args.method == '1':
        ga_main1()
    elif args.method == '2':
        ga_main2()


if __name__ == '__main__':
    main()
