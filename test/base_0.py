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

from eclib.benchmarks import rosenbrock, zdt1, zdt2, zdt3, zdt4, zdt6
from eclib.operations import UniformInitializer
from eclib.operations import RandomSelection
from eclib.operations import RouletteSelection
from eclib.operations import TournamentSelection
from eclib.operations import TournamentSelectionStrict
from eclib.operations import TournamentSelectionDCD
from eclib.operations import BlendCrossover
from eclib.operations import SimulatedBinaryCrossover
from eclib.operations import PolynomialMutation
from eclib.optimizers import NSGA2
from eclib.optimizers import MOEAD
from eclib.base import Individual
from eclib.base import NoisingIndividual

import myutils as ut


################################################################################

def identity(x):
    return x


def clip(x):
    return np.clip(x, 0.0, 1.0)


################################################################################

class NSGA2_ENV(object):
    def __init__(self, problem):
        # パラメータ
        n_dim = 30
        pop_size = 100
        # epoch = 250
        # save_trigger = lambda i: i == 1 or i % 10 == 0 # 10 epochごと
        # save_trigger = lambda i: i == epoch              # 最後だけ

        if problem == zdt4 or problem == zdt6:
            n_dim = 10
        if problem == zdt4:
            Individual.set_bounds([0.0] + [-5.0] * (n_dim - 1),
                                  [1.0] + [5.0] * (n_dim - 1))
        # Individual.set_bounds([0], [1])
        # Individual.set_weight([1, 1])

        # 問題
        # problem = zdt4
        initializer = UniformInitializer(n_dim)
        # selection = TournamentSelection(ksize=2)
        selection = TournamentSelectionStrict(ksize=2)
        # selection = TournamentSelectionDCD()
        # crossover = BlendCrossover(alpha=0.5)
        crossover = SimulatedBinaryCrossover(rate=0.9, eta=20)
        mutation = PolynomialMutation(rate=1/n_dim, eta=20)

        optimizer = NSGA2(pop_size, selection, crossover, mutation,
                          indiv_type=NoisingIndividual)
        # optimizer.set_initializer(Initializer(3))
        optimizer.setup(problem)

        ### Additional setting ###
        optimizer.initializer = initializer
        optimizer.n_cycle = None
        # optimizer.alternation = 'replace'
        ##########################

        self.optimizer = optimizer

    def __enter__(self):
        return self.optimizer

    def __exit__(self, exc_type, exc_value, traceback):
        self.optimizer.clear()


class MOEAD_ENV(object):
    def __init__(self, problem):
        # パラメータ
        n_dim = 30
        pop_size = 100
        ksize = 5 # MOEA/D近傍サイズ

        # epoch = 250
        # save_trigger = lambda i: i == 1 or i % 10 == 0 # 10 epochごと
        # save_trigger = lambda i: i == epoch              # 最後だけ

        if problem == zdt4 or problem == zdt6:
            n_dim = 10
        if problem == zdt4:
            Individual.set_bounds([0.0] + [-5.0] * (n_dim - 1),
                                  [1.0] + [5.0] * (n_dim - 1))
        # Individual.set_bounds([0], [1])
        # Individual.set_weight([1, 1])

        # 問題
        # problem = zdt4
        initializer = UniformInitializer(n_dim)
        selection = RandomSelection()
        # selection = TournamentSelection(ksize=2)
        # selection = TournamentSelectionStrict(ksize=2)
        # selection = TournamentSelectionDCD()
        # crossover = BlendCrossover(alpha=0.5)
        crossover = SimulatedBinaryCrossover(rate=0.9, eta=20)
        mutation = PolynomialMutation(rate=1/n_dim, eta=20)

        optimizer = MOEAD(pop_size, selection, crossover, mutation, ksize)
        # optimizer.set_initializer(Initializer(3))
        optimizer.setup(problem)

        ### Additional setting ###
        optimizer.initializer = initializer
        optimizer.n_cycle = None
        optimizer.alternation = 'all'
        ##########################

        self.optimizer = optimizer

    def __enter__(self):
        return self.optimizer

    def __exit__(self, exc_type, exc_value, traceback):
        self.optimizer.clear()


################################################################################
### TEMP ###
################################################################################

def get_optomal_front(file):
    import json
    with open(file, 'r') as optimal_front_data:
        optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    optimal_front = sorted(optimal_front[i]
                           for i in range(0, len(optimal_front), 2))
    return np.array(optimal_front)


def diversity(first_front, first, last):
    from math import hypot
    df = hypot(first_front[0][0] - first[0],
               first_front[0][1] - first[1])
    dl = hypot(first_front[-1][0] - last[0],
               first_front[-1][1] - last[1])
    dt = [hypot(f[0] - s[0],
                f[1] - s[1])
          for f, s in zip(first_front[:-1], first_front[1:])]

    if len(first_front) == 1:
        return df + dl

    dm = sum(dt)/len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di)/(df + dl + len(dt) * dm )
    return delta


def convergence(first_front, optimal_front):
    from math import sqrt
    distances = []

    for ind in first_front:
        distances.append(float("inf"))
        for opt_ind in optimal_front:
            dist = 0.
            for i in range(len(opt_ind)):
                dist += (ind[i] - opt_ind[i])**2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = sqrt(distances[-1])

    return sum(distances) / len(distances)


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
        print('init', args)

def __test__():
    print(np.array(np.array((0,))))


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
        ga_main1()
    elif args.method == '2':
        ga_main2()


if __name__ == '__main__':
    main()
