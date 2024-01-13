from tree.DecisionTreeClassifier import DecisionTreeClassifier
from utils import *


class RandomForestClassifier:

    """

    Attributes
    -------------

    n_estimators: int, optional (default=100)
    The number of trees

    Other arguments are explained in tree.DecisionTreeClassifier

    Methods
    -------------

    """

    def __init__(
            self,
            n_estimators=100,
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            max_features=None,
            class_weight=None,
            random_state=None,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = 'random'
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

        # the array with decision trees
        self.__trees = []
        self.__num_classes = 0

    def fit(self, X, y):
        self.__num_classes = len(np.unique(y))

        for _ in tqdm(range(self.n_estimators)):
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                splitter=self.splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            tree.fit(X, y)
            self.__trees.append(tree)

    def predict_proba(self, X):
        probs_sum = np.zeros(shape=(len(X), self.__num_classes))
        for tree in self.__trees:
            probs = tree.predict_proba(X)
            probs_sum += probs

        return probs_sum / self.n_estimators

    def predict(self, X):
        probs_X = self.predict_proba(X)
        return np.array([np.argmax(probs) for probs in probs_X])

