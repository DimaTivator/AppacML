import numpy as np
from _exceptions import ArgumentsShapeException


def accuracy_score(y_true: np.array, y_pred: np.array) -> float:
    if y_true.shape != y_pred.shape:
        raise ArgumentsShapeException(f"y_true shape {y_true.shape} doesn't match y_pred shape {y_pred.shape}")

    correct = np.count_nonzero((y_true - y_pred) == 0)
    shape = list(y_true.shape)
    total = ([shape[0]] + [shape[i] * shape[i - 1] for i in range(1, len(shape))])[-1]
    return correct / total
