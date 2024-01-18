import numpy as np
import matplotlib.pyplot as plt


def roc_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Generate a Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - y_true (numpy.ndarray): True binary labels (0 or 1).
    - y_prob (numpy.ndarray): Predicted probabilities for the positive class.

    Returns:
    None (plots the ROC curve).

    The function calculates True Positive Rate (TPR) and False Positive Rate (FPR)
    at various probability thresholds and plots the ROC curve.

    The implementation is naive and non-effective yet.
    """

    # TODO: provide effective implementation

    tpr_arr, fpr_arr = [], []

    for threshold in sorted(list(np.unique(y_prob)), reverse=True):
        y_pred = y_prob >= threshold

        true_positive = np.sum(y_true * y_pred)
        false_positive = np.sum((1 - y_true) * y_pred)
        false_negative = np.sum(y_true * (1 - y_pred))
        true_negative = np.sum((1 - y_true) * (1 - y_pred))

        TPR = true_positive / (true_positive + false_negative)
        FPR = false_positive / (false_positive + true_negative)

        tpr_arr.append(TPR)
        fpr_arr.append(FPR)

    plt.plot(fpr_arr, tpr_arr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()
