from utils import *


def unite(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack((X, y))


def train_test_split(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=None) -> tuple:
    """
    Split the input data into training and testing sets

    Parameters:
    - X (np.ndarray): Input features.
    - y (np.ndarray): Target variable.
    - test_size (float, optional): The proportion of the dataset to include in the test split.
      Should be between 0.0 and 1.0. Defaults to 0.2.
    - random_state (int, optional): Seed for random number generation. Defaults to None.
    """

    data = unite(X, y)

    random_generator = np.random.RandomState(random_state)
    indices = random_generator.permutation(np.array(list(range(X.shape[0]))))

    test_num_elements = math.ceil(X.shape[0] * test_size)
    test_indices = indices[:test_num_elements]
    train_indices = indices[test_num_elements:]

    data_train, data_test = data[train_indices], data[test_indices]

    sep = X.shape[1] - y.shape[1]

    X_train, X_test = data_train[:, :sep], data_test[:, :sep]
    y_train, y_test = data_train[:, sep:], data_test[:, sep:]

    return X_train, X_test, y_train, y_test
