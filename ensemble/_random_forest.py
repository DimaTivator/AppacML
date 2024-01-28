from tree import DecisionTreeClassifier
from utils import *
import tree

from preprocessing import train_test_split


class RandomForestClassifier(tree.Tree):

    """
    TODO: add docs

    Attributes
    -------------

    n_estimators: int, optional (default=100)
    The number of trees

    Other arguments are explained in tree.DecisionTreeClassifier

    Methods
    -------------

    fit(X, y)

    predict(X)

    predict_proba(X)

    """

    def __init__(
            self,
            n_estimators=100,
            criterion='entropy',
            splitter='best',
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            max_features=None,
            class_weight=None,
            random_state=None
    ):

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state
        )

        self.n_estimators = n_estimators

        # the array with decision trees
        self.__trees = []
        self.__num_classes = 0

        self.random_generator = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        self.__num_classes = len(np.unique(y))

        X = to_numpy(X)
        y = to_numpy(y)

        num_cols = X.shape[1] if len(X.shape) > 1 else X.shape[0]

        for _ in tqdm(range(self.n_estimators)):

            X_train, _, y_train, _ = train_test_split(
                X, y,
                test_size=0.7
            )

            cols = self.random_generator.choice(
                np.arange(num_cols),
                size=self.random_generator.randint(int(num_cols * 0.7), num_cols),
                replace=False
            )

            X_train = X_train[:, cols]

            self.features.append(cols)

            model = DecisionTreeClassifier(
                criterion=self.criterion,
                splitter=self.splitter,
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
        X = to_numpy(X)
        probs_sum = np.zeros(shape=(len(X), self.__num_classes))
        for i, model in enumerate(self.__trees):
            probs = model.predict_proba(X[:, self.features[i]])
            probs_sum += probs

        return probs_sum / self.n_estimators

    def predict(self, X):
        probs_X = self.predict_proba(X)
        return np.array([np.argmax(probs) for probs in probs_X])

    @property
    def trees(self):
        return self.__trees

