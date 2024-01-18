import numpy as np
from utils import *


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


def roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Receiver Operating Characteristic Area Under the Curve (ROC AUC) score.

    Parameters:
    - y_true (np.ndarray): 1D array of true binary labels (0 or 1).
    - y_pred (np.ndarray): 1D array of predicted probabilities or scores.

    Returns:
    - float: ROC AUC score, a value between 0 and 1. The higher the score, the better the model performance.

    The ROC AUC score is a performance metric for binary classification problems.
    It measures the area under the ROC curve, which is a graphical representation
    of the trade-off between true positive rate (sensitivity) and false positive rate.

    The input arrays y_true and y_pred should have the same length.

    In this implementation ROC-AUC score is computed as:
    {the number of pairs (i, j) where y_true[i] == 1 and y_true[j] == 0 and y_pred[i] > y_pred[j]}
    divided by
    {the number of pairs (i, j) where y_true[i] == 1 and y_true[j] == 0}

    """

    # if all predictions are the same, we assume that they are random
    if np.all(np.abs(y_pred - y_pred[0]) < EPS):
        return 0.5

    # Sort y_true based on the corresponding y_pred values
    y_true = y_true[np.argsort(y_pred)]

    n = len(y_true)
    cur_neg = 0
    pairs = 0
    for i in range(n):
        cur_neg += (y_true[i] == 0)
        pairs += cur_neg * y_true[i]

    score = pairs / (cur_neg * (n - cur_neg))
    return score
