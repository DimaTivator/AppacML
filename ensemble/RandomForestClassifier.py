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
            splitter='best',
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            max_features=None,
            class_weight=None,
            random_state=None,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        pass

    def predict_proba(self):
        pass

    def predict(self):
        pass

