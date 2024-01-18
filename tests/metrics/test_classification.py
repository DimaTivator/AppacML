import unittest
import numpy as np
from metrics import roc_auc_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score


class TestROCAUCScoreFunction(unittest.TestCase):

    def test_roc_auc_score_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 1.0])
        auc_score = roc_auc_score(y_true, y_pred)
        self.assertAlmostEqual(auc_score, 1.0, places=4)

    def test_roc_auc_score_all_predictions_same(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        auc_score = roc_auc_score(y_true, y_pred)
        self.assertAlmostEqual(auc_score, 0.5, places=4)

    def test_random_scores(self):
        # Test with random data
        np.random.seed(42)
        y_true = np.random.randint(2, size=100)
        y_pred = np.random.rand(100)

        your_score = roc_auc_score(y_true, y_pred)
        sklearn_score = sklearn_roc_auc_score(y_true, y_pred)

        self.assertAlmostEqual(your_score, sklearn_score, places=5)

    def test_perfect_classifier(self):
        # Test with a perfect classifier
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        your_score = roc_auc_score(y_true, y_pred)
        sklearn_score = sklearn_roc_auc_score(y_true, y_pred)

        self.assertAlmostEqual(your_score, sklearn_score, places=5)

    def test_worst_classifier(self):
        # Test with the worst possible classifier
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])

        your_score = roc_auc_score(y_true, y_pred)
        sklearn_score = sklearn_roc_auc_score(y_true, y_pred)

        self.assertAlmostEqual(your_score, sklearn_score, places=5)


if __name__ == '__main__':
    unittest.main()
