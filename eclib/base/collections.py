
class Collection(object):
    def __init__(self):
        self.data = []

    def __call__(self):
        return self.data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        for v in self.data:
            yield v

    def __len__(self):
        return len(self.data)


class Container(object):
    def __init__(self):
        self.value = []

    def __call__(self):
        return self.value

    def __getitem__(self, key):
        return self.value[key]

    def __iter__(self):
        for v in self.value:
            yield v

    def __len__(self):
        return len(self.value)

