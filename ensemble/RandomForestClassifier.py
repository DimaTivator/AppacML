from tree.DecisionTreeClassifier import DecisionTreeClassifier
from utils import *
import tree.tree as tree

from sklearn.model_selection import train_test_split


class RandomForestClassifier(tree.Tree):

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
            random_state=None
    ):

        super().__init__(
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state
        )

        self.n_estimators = n_estimators
        self.__splitter = 'random'

        # the array with decision trees
        self.__trees = []
        self.__num_classes = 0

        self.random_generator = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        self.__num_classes = len(np.unique(y))

        X = to_numpy(X)
        y = to_numpy(y)

        for _ in tqdm(range(self.n_estimators)):

            # TODO replace sklearn function with mine
            X_train, _, y_train, _ = train_test_split(
                X, y,
                test_size=0.7,
                random_state=self.random_generator.randint(0, 1000)
            )

            model = DecisionTreeClassifier(
                criterion=self.criterion,
                splitter=self.__splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)
            self.__trees.append(model)

    def predict_proba(self, X):
        probs_sum = np.zeros(shape=(len(X), self.__num_classes))
        for model in self.__trees:
            probs = model.predict_proba(X)
            probs_sum += probs

        return probs_sum / self.n_estimators

    def predict(self, X):
        probs_X = self.predict_proba(X)
        return np.array([np.argmax(probs) for probs in probs_X])

