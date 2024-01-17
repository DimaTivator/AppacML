import unittest
import numpy as np
from preprocessing import (
    train_test_split
)


class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        # Set up sample data for testing
        np.random.seed(42)
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.y = np.array([[0], [1], [0], [1]])

    def test_train_test_split_default(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(y_train.shape[0], 3)
        self.assertEqual(y_test.shape[0], 1)

    def test_train_test_split_custom_test_size(self):
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(y_train.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)

    def test_train_test_split_random_state(self):
        random_state = 10
        X_train1, X_test1, y_train1, y_test1 = train_test_split(self.X, self.y, random_state=random_state)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(self.X, self.y, random_state=random_state)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


if __name__ == '__main__':
    unittest.main()
