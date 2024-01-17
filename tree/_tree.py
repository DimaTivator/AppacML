from utils import *
from base import BaseEstimator
from abc import ABC


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
                f'Probs: {self.probs}\n'
                )


class Tree(BaseEstimator, ABC):

    def __init__(
            self,
            criterion='entropy',
            splitter='best',
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            max_features=None,
            class_weight=None,
            random_state=None,
    ):

        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state


