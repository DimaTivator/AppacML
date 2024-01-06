from utils import *


class Node:
    def __init__(
            self,
            X,
            y,
            depth,
            sep_feature,
            value,
            is_leaf=False,
            ID=0
    ):
        self.X = X
        self.y = y
        self.depth = depth
        self.sep_feature = sep_feature
        self.value = value
        self.is_leaf = is_leaf
        self.children = []
        self.probs = []
        self.__id = ID

    @property
    def ID(self):
        return self.__id

    def __str__(self):
        return (
                f'NODE ID: {self.__id}\n'
                f'Depth: {self.depth}\n'
                f'Value {self.value}\n'
                f'Sep by: feature {self.sep_feature}\n'
                f'Data: {np.concatenate((self.X, self.y), axis=1)}\n'
                )

