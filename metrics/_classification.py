import numpy as np
from _exceptions import ArgumentsShapeException


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = np.count_nonzero((y_true - y_pred) == 0)
    shape = list(y_true.shape)
    total = ([shape[0]] + [shape[i] * shape[i - 1] for i in range(1, len(shape))])[-1]
    return correct / total


# What part of retrieved items is relevant
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = np.sum(y_true * y_pred)    # the number of cases where 1 was correctly predicted
    false_positives = np.sum(~y_true * y_pred)  # the number of cases where 1 was incorrectly predicted
    return true_positives / (true_positives + false_positives)


# What part of relevant items is retrieved?
def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = np.sum(y_true * y_pred)  # the number of cases where 1 was correctly predicted
    all_positive = np.sum(y_true)
    return true_positive / all_positive


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
