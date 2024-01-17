import numpy as np
from _exceptions import ArgumentsShapeException


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ArgumentsShapeException(f"y_true shape {y_true.shape} doesn't match y_pred shape {y_pred.shape}")

    correct = np.count_nonzero((y_true - y_pred) == 0)
    shape = list(y_true.shape)
    total = ([shape[0]] + [shape[i] * shape[i - 1] for i in range(1, len(shape))])[-1]
    return correct / total


# What part of retrieved items is relevant
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = np.sum(y_true * y_pred)    # the number of cases where 1 was correctly predicted
    false_positives = np.sum(~y_true * y_pred)  # the number of cases where 1 was incorrectly predicted
    return true_positives / (true_positives + false_positives)
