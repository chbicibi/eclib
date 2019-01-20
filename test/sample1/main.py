import argparse
import glob
import os
import sys
import subprocess
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from matplotlib import rcParams

import model as M
import vis as V
import utils as ut


def ga_main1(method_name, out='result', clear_directory=False):
    ''' GA実行 '''

    if clear_directory and os.path.isdir(out):
        shutil.rmtree(out)

    popsize = 100
    epoch = 100
    ksize = 5

    # GA環境
    with M.Optimize_ENV(method_name, popsize=popsize, ksize=ksize) as env:
        optimizer = env.optimizer
        creator = env.creator
        with ut.stopwatch('main'):
            # GA開始
            # 初期集団生成
            population = optimizer.init_population(creator, popsize=popsize)
            history = [population]
            print(history[0][0].data.value)
            return

            # 進化
            for i in range(1, epoch + 1):
                population = optimizer(population)
                history.append(population)

                print('epoch:', i, 'popsize:', len(population), end='\r')
                if i == epoch:
                    # モデルをファイルに書き込み
                    file = f'popsize{popsize}_epoch{i}_{ut.strnow("%Y%m%d_%H%M")}.pkl'
                    file = os.path.join(out, file)
                    print('save:', file)
                    # optimizer.save(file=os.path.join(out, file))
                    ut.save(file, history)
            return history


def ga_plot1(out='result'):
    ''' 解の親子関係を線で結んだアニメーションを作成 '''

    def get_model():
        ''' モデル読み込み '''
        # model_cls = {'nsga2':NSGA2, 'moead':MOEAD}[model]
        files = ut.fsort(glob.glob(os.path.join(out, f'*epoch*.pkl')))
        print('select file')
        for i, file in enumerate(files):
            print(f'[{i}]', file)
        print(f'[{len(files)}] Exit')
        try:
            file = files[int(input())]
        except (ValueError, IndexError):
            return
        print('file:', file)
        history = ut.load(file)
        return history

    def resume_main(history):
        ''' プロット(表示) '''
        print('resume_main')
        fig, ax = plt.subplots(figsize=(8, 8))
        # with plt.xkcd():
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.99)
        plot = V.get_plot(fig, ax)

        # ax.tick_params(labelbottom=False, bottom=False) # x軸の削除
        # ax.tick_params(labelleft=False, left=False) # y軸の削除
        os.makedirs('result/img0', exist_ok=True)

        try:
            def f_():
                for i, population in enumerate(history):
                    for _ in plot(population):
                        yield

            with ut.chdir('result/img0'):
                for i, _ in enumerate(f_()):
                    if i > 298:
                        break
                    # plt.pause(0.3)
                    plt.savefig(f'anim_{i:05d}.png')
                    # origin = population[0].data.origin.origin or []
                    # print([x.id for x in origin], '->', population[0].data.id)
                subprocess.run(['ffmpeg', '-y', '-r', '16', '-i', 'anim_%05d.png',
                                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                                '-s', '800x800', '-r', '16', '../anim0.mp4'])
            # plt.show()

        except KeyboardInterrupt:
            return

    def resume_anim(history):
        ''' プロット(アニメーション出力) '''
        print('resume_anim')
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.99)
        plot = V.get_plot(fig, ax)
        # def init_func():
        #     ax.tick_params(labelbottom=False, bottom=False) # x軸の削除
        #     ax.tick_params(labelleft=False, left=False) # y軸の削除
        D = {'it':None}
        def update(i):
            print(i, end='\r')
            if not D['it']:
                population = history.pop(0)
                D['it'] = plot(population)
            try:
                next(D['it'])
            except StopIteration:
                D['it'] = None
                update(i)
            # for i, population in enumerate(history):
                # yield
        show_anim(fig, update, file='result/anim.mp4', frames=299, fps=16)

    history = get_model()
    if history:
        resume_main(history)
        # resume_anim(history)


################################################################################

FFMpegWriter = None
def init_ffmpeg():
    global FFMpegWriter
    bins_ffmpeg = glob.glob('../../**/ffmpeg.exe', recursive=True)
    if not bins_ffmpeg:
        return
    rcParams["animation.ffmpeg_path"] = bins_ffmpeg[0]
    FFMpegWriter = anim.writers['ffmpeg']
    return FFMpegWriter


def show_anim(fig, update, frames=1000, init_func=lambda:None, interval=8, file=None, fps=2):
  ani = anim.FuncAnimation(fig, update, frames=frames, init_func=init_func, interval=interval)
  if file:
    ani.save(file, writer=FFMpegWriter(fps=fps))
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


def pltenv():
    rcParams['figure.figsize'] = 11, 7 # default=>(6.4, 4.8)
    rcParams['font.family'] = 'Times New Roman', 'serif'
    rcParams['font.size'] = 24 # default=>10.0

    # rcParams["mathtext.rm"] = 'Times New Roman'
    # rcParams["mathtext.it"] = 'Times New Roman'
    # rcParams["mathtext.bf"] = 'Times New Roman'
    # rcParams["mathtext.rm"] = 'Times New Roman'
    # rcParams["mathtext.sf"] = 'Times New Roman'
    # rcParams["mathtext.tt"] = 'Times New Roman'

    rcParams['savefig.directory'] = 'data'
    rcParams['savefig.transparent'] = True
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
    elif args.method == 'p1':
        ga_plot1(out=out)



if __name__ == '__main__':
    main()

'''
Note:
2018.11.21 08:22
MOEA/D ksize=50 N=500 # ksize大きくすると多様性が減少
2018.11.21 08:25
MOEA/D ksize=20 N=500 # ksize大きくすると多様性が減少

'''
