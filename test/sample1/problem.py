import ctypes
import random
import numpy as np


################################################################################

def get_f_initialize(cdll):
    ''' ラッパー1: initialize
    '''
    func_ptr = getattr(cdll, 'initialize')
    func_ptr.argtypes = []
    func_ptr.restype = ctypes.c_void_p
    def f_():
        func_ptr()
    return f_


def get_f_init_debri(cdll):
    ''' ラッパー2: init_debri
    '''
    func_ptr = getattr(cdll, 'init_debri')
    func_ptr.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                         ctypes.POINTER(ctypes.c_int32)]
    func_ptr.restype = ctypes.c_void_p
    def f_(tle_file, rcs_file):
        arg_ptr0 = ctypes.c_char_p(tle_file.encode())
        arg_ptr1 = ctypes.c_char_p(rcs_file.encode())
        n_debris = ctypes.c_int32()
        func_ptr(arg_ptr0, arg_ptr1, ctypes.byref(n_debris))
        return n_debris.value
    return f_


def get_f_call_problem(cdll):
    ''' ラッパー3: call_problem
    '''
    func_ptr = getattr(cdll, 'call_problem')
    func_ptr.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64),
    ]
    func_ptr.restype = ctypes.c_void_p
    result = np.zeros(2, dtype=np.float64)
    def f_(iargs, dargs):
        n1 = ctypes.c_int32(iargs.size)
        n2 = ctypes.c_int32(dargs.size)
        func_ptr(iargs, n1, dargs, n2, result)
        return result
    return f_


def __test__():
    # CDLLインスタンス作成
    libname = 'case0.dll'
    loader_path = '.'
    cdll = np.ctypeslib.load_library(libname, loader_path)

    # 関数取得
    f_initialize = get_f_initialize(cdll)
    f_init_debri = get_f_init_debri(cdll)
    f_call_problem = get_f_call_problem(cdll)

    # 初期化: 開始時刻設定
    f_initialize()

    # 初期化: デブリデータ読み込み
    tle_file = '../data/debri_elements.txt'
    rcs_file = '../data/RCS_list.txt'
    n_debris = f_init_debri(tle_file, rcs_file)
    print('n_debris:', n_debris)

    # 関数呼び出し
    for i in range(1):
        a = np.array([0, 2, 3, 4, 5], dtype=np.int32)
        b = np.array([0, 1800, 0, 1800, 0, 1800, 0, 1800], dtype=np.float64)
        delv, rcs = f_call_problem(a, b)
    print(delv, rcs)


################################################################################


if __name__ == '__main__':
    __test__()
    # print(np.empty_like(np.array([1.0])))
