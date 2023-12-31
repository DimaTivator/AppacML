from utils import *


def entropy(array):
    classes = np.unique(array)
    n = len(array)
    return -sum([np.count_nonzero(array == c) / n * np.log2(np.count_nonzero(array == c) / n)
                 for c in classes])


def gini(array):
    classes = np.unique(array)
    n = len(array)
    return 1 - sum([(np.count_nonzero(array == c) / n) ** 2 for c in classes])


