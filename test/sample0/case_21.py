#! /usr/bin/env python3

'''
多目的最適化アルゴリズムのテスト
最小機能版
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

def get_result_file(root='.'):
    ''' 計算済み最適化ファイルを読み出す
    '''
    with ut.chdir(root):
        files = glob.glob('optimize_*.pkl')
        for i, file in enumerate(files):
            print(f'[{i}] {file}')
        print('select file')
        n = int(input())
        if n < 0:
            return
        file = files[n]
        print('file:', file)
        env, optimizer, history = ut.load(file)
        return env, optimizer, history


################################################################################

def main1(out='result', force=False, clear_directory=False):
    ''' 最適化実行(ミニマル)
    '''
    ''' ======== 最適化パラメータの定義 ======== '''
    problem = zdt1
    n_dim = 30
    popsize = 100
    epoch = 200

    with Environment() as env:
        ''' ======== 必要なオブジェクトの準備 ======== '''
        # 個体管理クラス
        indiv_pool = env.register(Individual)

        # 遺伝子生成クラス
        initializer = UniformInitializer(n_dim)

        # 初期個体生成クラス
        creator = Creator(initializer, indiv_pool)

        # 最適化オペレータ
        op0 = NSGA2(problem=problem, pool=indiv_pool)
        op1 = MOEAD(problem=problem, pool=indiv_pool, ksize=5)
        optimizer = op0

        ''' ======== 初期集団の作成 ======== '''
        population = optimizer.init_population(creator, popsize=popsize)
        history = [population]

        ''' ======== 最適化 ======== '''
        for i in range(epoch):
            print('epoch', i + 1)
            population = optimizer(population)
            history.append(population)

            if i % 50 == 50-1:
                ut.save(f'result/temp{i}.pkl', history)

            if i >= epoch // 2:
                optimizer = op1

        ''' ======== 結果保存 ======== '''
        data = env, optimizer, history
        ut.save(f'result/optimize_#{ut.snow}.pkl', data)
        return env, optimizer, history


def main2(out='result', force=False, clear_directory=False):
    ''' 親子関係のプロット
    '''
    def line(ind, origin):
        if not origin:
            return []
        return list(zip(origin[0].value, ind.value, origin[1].value))
    def dist(ind, origin):
        ''' 親子間の距離の合計値を計算 '''
        if not origin:
            return 0
        values = np.array([[ind.value, par.value] for par in origin])
        diff = values[:, :, 0] - values[:, :, 1]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return np.sum(dists)

    fig, ax = plt.subplots()
    cm = plt.get_cmap('jet')
    def plot(pop, message):
        # 親子距離計算
        pairs = [(fit.data, fit.data.origin.origin or ()) for fit in pop]
        parents = list(chain(*(x[1] for x in pairs)))
        lines = [line(ind, origin) for ind, origin in pairs]
        # print(lines)
        # exit()
        if lines and [x for x in lines if x]:
            dists = [dist(ind, origin) for ind, origin in pairs]
            print('dist(sum)=', sum(dists))
        else:
            print()

        # プロット
        ax.cla()
        if parents:
            x_p, y_p = np.array([ind.value for ind in parents]).T
            ax.scatter(x_p, y_p, c='r')

            for i, l in enumerate(lines):
                if l:
                    ax.plot(*l, c=cm(i/(len(lines)+1)), linewidth=0.5)

        x, y = np.array([fit.data.value for fit in pop]).T
        # ax.set_xlim(0, 2)
        # ax.set_ylim(0, 2)
        ax.scatter(x, y, c='b')
        ax.annotate(message, xy=(0.95, 0.01),
                    xycoords='figure fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=12)
        plt.pause(1e-3)

    data = get_result_file('result')
    if not data:
        return
    env, optimizer, history = data

    for i, population in enumerate(history):
        print(f'epoch{i+1}:', end=' ')
        message = f'epoch{i+1}: {population.origin.name}'
        plot(population, message)
    plt.show()


################################################################################

def __test__():
    print(globals())


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

    if args.method in '0123456789':
        name = 'main' + str(args.method)
        D = globals()
        if name in D:
            D[name](out=out, force=args.force)


if __name__ == '__main__':
    main()
