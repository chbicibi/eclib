# from .collections import Collection


class Population(object):
    ''' 解集団
    GAの個体を管理する (デフォルトではシーケンス型)
    島モデルの場合はmigrationを担当する
    '''

    def __init__(self, data=None, capacity=None):
        super().__init__()
        if isinstance(data, Population):
            self.data = data.data
            self.capacity = capacity or data.capacity
        else:
            self.data = data or []
            self.capacity = capacity

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return self.data + other.data

    def append(self, x):
        self.data.append(x)

    def clear(self):
        self.data.clear()

    def filled(self):
        return self.capacity and len(self.data) >= self.capacity
