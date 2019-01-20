import argparse
import glob
import os
import sys
from itertools import chain
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from matplotlib import rcParams

import utils as ut


def sigmoid(v, a):
    return 1/(1+np.exp(-a*v))


def fix_(v):
    # return np.array([v[0], sigmoid(v[1], 0.5)])
    return np.array(list(v))


def len_(x, y):
    r = 1, 3 # plot range
    return sqrt(sum(((v0 - v1) / v2) ** 2 for v0, v1, v2 in zip(x, y, r)))


def mk_line(ind, origin, no):
    '''  親1→子→親2の座標を返す '''
    if not origin:
        return []

    v_ind = fix_(ind.value)
    vs_ori = map(fix_, origin)

    # return list(zip(origin[0].value, ind.value, origin[1].value))
    data = ((v_ind, v_ori, len_(v_ind, v_ori)) for v_ori in vs_ori)
    return (dict(zip(('head', 'tail', 'len', 'no'), d+(no,))) for d in data)


def get_plot(fig, ax, it=True):
    ''' 解の親子関係を線で結んだアニメーションを作成 '''
    ymax = float('inf')
    arrowprops_od = dict(edgecolor='blue', headwidth=30, width=15)
                         # arrowstyle='<|-, head_width=0.5',lw=3)
    rcParams['savefig.transparent'] = False

    def plot_texts():
        ax.annotate(s='Optimum Direction', xy=(0.2, 0.05),
                    xycoords='axes fraction')
        ax.annotate(s='', xy=(0.1, 0.1), xytext=(0.2, 0.2),
                    xycoords='axes fraction', arrowprops=arrowprops_od)
        ax.annotate(s='Objective1', xy=(0.5, -0.1), xycoords='axes fraction',
                    horizontalalignment="center", verticalalignment="center")
        ax.annotate(s='Objective2', xy=(-0.135, 0.5), xycoords='axes fraction',
                    horizontalalignment="center", verticalalignment="center",
                    rotation=90)

    def plot_func(pop):
        pairs = [(fit.data, fit.data.origin.origin or (), i)
                 for i, fit in enumerate(pop)]
        parents = list(chain(*(x[1] for x in pairs)))
        lines = list(chain(*(mk_line(*ps) for ps in pairs)))
        n_pop = len(pop)

        ax.cla()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 3))

        cm = plt.get_cmap('jet')
        # print(cm(10))
        # exit()
        arrowprops = lambda c, l: dict(edgecolor=c, facecolor='white', width=0.1, headwidth=5.0,
                                    headlength=5.0, shrink=0.01/l)

        if parents:
            x_p, y_p = np.array([fix_(ind.value) for ind in parents]).T
            ax.scatter(x_p, y_p, s=40, c='b')

            for i, l in enumerate(lines):
                if l['len'] > 0.05:
                    try:
                        color = cm(l['no']/(n_pop+1))
                        # ax.plot(*l, c=cm(i/(len(lines)+1)), linewidth=0.5)
                        # ax.arrow(**l, width=0.01, head_width=0.05, head_length=0.2,
                        #          length_includes_head=True, color='k')
                        ax.annotate(s='', xy=l['head'], xytext=l['tail'],
                                    xycoords='data',
                                    arrowprops=arrowprops(color, l['len']))
                    except:
                        print(l)
                        raise

        x, y = np.array([fix_(fit.data.value) for fit in pop]).T
        ax.scatter(x, y, s=80, c='pink', alpha=0.5, linewidths=20, edgecolors='red')
        # plt.pause(1e-10)
        # ax.annotate("MOEA/D", xy=(0.5, -0.08), xycoords="axes fraction", fontsize=28, horizontalalignment="center", verticalalignment="top")

    def plot_func_it(pop):
        nonlocal ymax

        pairs = [(fit.data, fit.data.origin.origin or (), i)
                 for i, fit in enumerate(pop)]
        parents = list(chain(*(x[1] for x in pairs)))
        lines = list(chain(*(mk_line(*ps) for ps in pairs)))
        n_pop = len(pop)

        ymax = min(max(*(fit.data.value[1] for fit in pop), 1), ymax)

        ax.cla()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 5))
        plot_texts()

        cm = plt.get_cmap('jet')
        # print(cm(10))
        # exit()
        arrowprops = lambda c, l: dict(edgecolor=c, facecolor='white',
                                       width=0.5, linewidth=0.5, headwidth=5.0,
                                       headlength=5.0, shrink=0.01/l, alpha=0.3)

        if parents:
            # PLOT[0]
            x_p, y_p = np.array([fix_(ind.value) for ind in parents]).T
            ax.scatter(x_p, y_p, s=140, c='pink', alpha=0.8, linewidths=2,
                       edgecolors='red')
            yield

            ax.cla()
            ax.set_xlim((0, 1))
            ax.set_ylim((0, 5))
            plot_texts()

            # PLOT[1]
            ax.scatter(x_p, y_p, s=70, c='lightblue', alpha=0.5, linewidths=1.5,
                       edgecolors='blue')

            for i, l in enumerate(lines):
                if l['len'] > 0.05 and l['tail'][1] < ymax:
                    try:
                        color = cm(l['no']/(n_pop+1))
                        ax.annotate(s='', xy=l['head'], xytext=l['tail'],
                                    xycoords='data',
                                    arrowprops=arrowprops(color, l['len']))
                    except:
                        print(l)
                        raise
            yield

        # PLOT[3]
        x, y = np.array([fix_(fit.data.value) for fit in pop]).T
        ax.scatter(x, y, s=140, c='pink', alpha=0.8, linewidths=2,
                   edgecolors='red')
        yield

    if it:
        return plot_func_it # ジェネレータバージョン
    else:
        return plot_func # 関数バージョン


################################################################################

def ffmpegwriter(cache=[], *args, **kwargs):
    if cache:
        return cache[0](*args, **kwargs)

    bins_ffmpeg = glob.glob('../../**/ffmpeg.exe', recursive=True)
    if not bins_ffmpeg:
        raise FileNotFoundError

    rcParams['animation.ffmpeg_path'] = bins_ffmpeg[0]
    writer = anim.writers['ffmpeg']
    cache.append(writer)
    return writer(*args, **kwargs)


def show_anim(fig, update, frames=1000, init_func=lambda:None, interval=8,
              file=None, fps=2):
  ani = anim.FuncAnimation(fig, update, frames=frames, init_func=init_func,
                           interval=interval)
  if file:
    ani.save(file, writer=ffmpegwriter(fps=fps))
  else:
    plt.show()


################################################################################

def __test__():
    libname = 'problem.dll'
    loader_path = 'fortran'
    # cdll = np.ctypeslib.load_library(libname, loader_path)
    cdll = ctypes.WinDLL(libname)

    f_initialize = orbitlib.get_f_initialize(cdll)
    f_init_debri = orbitlib.get_f_init_debri(cdll)
    f_call_problem = orbitlib.get_f_call_problem(cdll)
    f_initialize()
    print(cdll)


def __test__():
    print(1/(1+exp(-0.1)))


def pltenv():
    rcParams['figure.figsize'] = 11, 11 # default=>(6.4, 4.8)
    rcParams['font.family'] = 'Times New Roman', 'serif'
    rcParams['font.size'] = 24 # default=>10.0

    # rcParams["mathtext.rm"] = 'Times New Roman'
    # rcParams["mathtext.it"] = 'Times New Roman'
    # rcParams["mathtext.bf"] = 'Times New Roman'
    # rcParams["mathtext.rm"] = 'Times New Roman'
    # rcParams["mathtext.sf"] = 'Times New Roman'
    # rcParams["mathtext.tt"] = 'Times New Roman'

    rcParams['savefig.directory'] = 'data'
    rcParams['savefig.transparent'] = False
    init_ffmpeg()


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='?', default='',
                        help='Main method type')
    parser.add_argument('--model', '-m', default='nsga2',
                        help='Model type')
    parser.add_argument('--out', '-o', default='',
                        help='Output directory')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove output directory before start')
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
    model = args.model
    out = os.path.join('result', args.out)

    if args.test:
        __test__()
        return

    pltenv()

    if args.method == 'm1':
        ga_main1(model, out=out, clear_directory=args.clear)
    elif args.method == 'r1':
        ga_res_temp1(out=out)



if __name__ == '__main__':
    main()

'''
Note:
2018.11.21 08:22
MOEA/D ksize=50 N=500 # ksize大きくすると多様性が減少
2018.11.21 08:25
MOEA/D ksize=20 N=500 # ksize大きくすると多様性が減少

'''
