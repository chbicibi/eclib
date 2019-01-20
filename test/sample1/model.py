import argparse
import ctypes
import glob
import os
import shutil
import sys
from itertools import chain
from operator import attrgetter, itemgetter

import numpy as np

from eclib.base import Individual
from eclib.base import Environment
from eclib.base import Creator
from eclib.operations import UniformInitializer
from eclib.operations import TournamentSelection
from eclib.operations import SimulatedBinaryCrossover
from eclib.operations import PolynomialMutation
from eclib.optimizers import NSGA2
from eclib.optimizers import MOEAD

# テスト関数
from eclib.benchmarks import rosenbrock, zdt1, zdt2, zdt3, zdt4, zdt6

import utils as ut
import problem as orbitlib


################################################################################

FUNC_FORTRAN = None
class Problem(object):
    def __init__(self, size=0):
        global FUNC_FORTRAN

        if not FUNC_FORTRAN:
            # CDLLインスタンス作成
            libname = 'case0.dll'
            loader_path = 'fortran'
            cdll = np.ctypeslib.load_library(libname, loader_path)
            # cdll = ctypes.WinDLL(libname)

            # 関数取得
            f_initialize = orbitlib.get_f_initialize(cdll)
            f_init_debri = orbitlib.get_f_init_debri(cdll)
            f_call_problem = orbitlib.get_f_call_problem(cdll)

            # 初期化: 開始時刻設定
            f_initialize()

            # 初期化: デブリデータ読み込み
            tle_file = '../data/debri_elements.txt'
            rcs_file = '../data/RCS_list.txt'
            n_debris = f_init_debri(tle_file, rcs_file)
            print('n_debris:', n_debris)
            FUNC_FORTRAN = f_call_problem, n_debris
        else:
            f_call_problem, n_debris = FUNC_FORTRAN

        self._size = size
        self.function = f_call_problem
        self.imax = n_debris

    def __call__(self, genome):
        # 関数呼び出し
        # order = genome.get_ivalue()
        # params = genome.get_dvalue()
        # delv, rcs = self.function(order, params)
        value = (sum(genome), sum(genome ** 2))
        print(f'objF(test): args=', genome, 'value=', value)
        return value

    def __reduce_ex__(self, protocol):
        return type(self), (self._size,)


def manual_problem(array):
    obj0 = sum(array)
    obj1 = 1 / obj0
    return obj0, obj1



################################################################################

class Optimize_ENV(object):
    def __init__(self, method, popsize=100, **kwargs):
        method = method.lower()
        if 'nsga' in method:
            opt_cls = NSGA2
            ga_ops = {}
        elif 'moea' in method:
            opt_cls = MOEAD
            if kwargs['ksize']:
                ksize = kwargs['ksize']
            else:
                ksize = 5
            ga_ops = {'ksize': ksize}
        else:
            print('Unknown method name:', method)
            raise RuntimeError

        # 設計変数の次元数
        n_dim = 30

        # 設計範囲
        low_bounds = 0
        upp_bounds = 1

        # 最適化重み(正=>最小化, 負=>最大化)
        opt_weight = [1, 1]

        # 問題関数
        # problem = Problem()
        problem = zdt2

        with Environment() as env:
            # 個体クラス
            indiv_pool = env.register(Individual)

            # 遺伝子初期化クラス
            initializer = UniformInitializer(n_dim)

            # GAオペレータ指定
            ga_ops = {
              'selection': TournamentSelection(ksize=2),
              'crossover': SimulatedBinaryCrossover(rate=0.9, eta=20),
              'mutation': PolynomialMutation(rate=0.1, eta=20),
              **ga_ops
            }

            # GAクラス
            optimizer = opt_cls(problem=problem, pool=indiv_pool, **ga_ops)

            ### Additional setting ###
            # 設計範囲設定
            indiv_pool.cls.set_bounds(low_bounds, upp_bounds) # (下限，上限)

            # 最適化重み付け(正=>最小化, 負=>最大化)
            indiv_pool.cls.set_weight(opt_weight)

            # 親個体選択時の初期化周期
            optimizer.n_cycle = popsize // 2
            ##########################

            # 個体生成器
            creator = Creator(initializer, indiv_pool)

            # 登録
            env.optimizer = optimizer
            env.creator = creator
            self.env = env

    def __enter__(self):
        return self.env

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        # self.optimizer.clear()


################################################################################

def __test__():
    libname = 'case0.dll'
    loader_path = '.'
    # cdll = np.ctypeslib.load_library(libname, loader_path)
    cdll = ctypes.WinDLL(libname)

    f_initialize = orbitlib.get_f_initialize(cdll)
    f_init_debri = orbitlib.get_f_init_debri(cdll)
    f_call_problem = orbitlib.get_f_call_problem(cdll)
    f_initialize()
    print(cdll)


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():

    if args.test:
        __test__()
        return


if __name__ == '__main__':
    main()
