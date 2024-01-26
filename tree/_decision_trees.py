from preprocessing import EqualLengthBinner
from utils.utils import *
from metrics import criterion as crit
import tree


class DecisionTreeClassifier(tree.Tree):
    """
    ! It is assumed that the class labels are numbered from 0 to n,
     where n is the total number of classes


    Attributes
    -------------

    criterion: str, optional (default='entropy')
        The function to measure the quality of split
        {'gini', 'entropy'}

    splitter: str, optional (default='best')
        Split strategy
        {'best', 'random'}

    min_samples_split: int, optional (default=2)
        The minimum number of samples to split a node by a taken feature

    min_samples_leaf: int, optional (default=1)
        The minimum number of samples required to be at a leaf node

    max_depth: int or None, optional (default=None)
        The maximum depth of the tree. If it's reached, this branch
        stops building

    max_features: int, float, str or None, optional (default=None)
        The number of features to consider when looking for the best split.
        If None, then consider all features.

    class_weight: dict, list of dict or "balanced", optional (default=None)
        Weights associated with classes in the form {class_label: weight}.
        If None, all classes have equal weight. If "balanced", class weights
        are computed based on the input data.

    random_state: int, RandomState instance or None, optional (default=None)
        Controls the randomness of the estimator. Pass an int for reproducible
        results across multiple function calls.


    Methods
    -------------

    fit(X, y)
        X, y: pandas.DataFrame | pandas.Series | numpy.ndarray | list
        Builds a decision tree. Building algorithm -- ID3

    predict(X)

    predict_proba(X)

    print_tree()
    """

    def __init__(
            self,
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

        self.criterion_dict = {
            'gini': crit.gini,
            'entropy': crit.entropy
        }

        self.random_generator = np.random.RandomState(self.random_state)
        self.__root = None

        self.__binners = []

    def fit(self,
            X: pd.DataFrame | pd.Series | np.ndarray,
            y: pd.DataFrame | pd.Series | np.ndarray
            ):

        id_gen = int_gen()

# ----------------------------------------------------------------------------------

        def is_terminal(node: tree.Node):
            return (
                    node.depth == self.max_depth or
                    len(np.unique(node.y)) == 1
            )

        def get_split(node: tree.Node, idx: int) -> list:
            col = node.X[:, idx]
            return [node.y[float_eq(node.X[:, idx], value)] for value in np.unique(col)]

        def calc_probs(node: tree.Node):
            n = len(node.y)  # the total number of rows
            probs = np.zeros(num_classes)

            for i, c in enumerate(list(range(num_classes))):
                probs[i] = np.count_nonzero(node.y == c) / n

            node.probs = probs

            for child in node.children:
                calc_probs(child)

        # returns the index of separating feature
        def best_split(node: tree.Node) -> int:
            min_score = INF
            sep_feature_idx = -1

            min_usage = min(use_count)

            features_idx = [i for i in range(len(node.X[0])) if use_count[i] == min_usage]

            # best split algorithm
            # iterates through all features that have been used the minimum number of times
            for i in features_idx:
                s_split = get_split(node, i)

                split_score = sum([self.criterion_dict[self.criterion](s_split[i])
                                   for i in range(len(s_split))])

                if split_score < min_score and len(s_split) >= self.min_samples_split:
                    min_score = split_score
                    sep_feature_idx = i

            if sep_feature_idx != -1:
                use_count[sep_feature_idx] += 1

            return sep_feature_idx

        def random_split(node: tree.Node) -> int:
            # # getting the indices of features that can split data
            # # on more than min_samples_split samples
            # # and have been used the minimum number of times
            # min_usage = min(use_count)
            # features_idx = []
            # for i in range(len(node.X[0])):
            #     s_split = get_split(node, i)
            #     if len(s_split) >= self.min_samples_split and use_count[i] == min_usage:
            #         features_idx.append(i)

            # getting the indices of features that can split data
            # on more than min_samples_split samples
            features_idx = []
            for i in range(len(X[0])):
                s_split = get_split(node, i)
                if len(s_split) >= self.min_samples_split:
                    features_idx.append(i)

            return self.random_generator.choice(features_idx) if len(features_idx) > 0 else -1

        def build(node: tree.Node):
            sep_feature_idx = split_dict[self.splitter](node)

            if is_terminal(node) or sep_feature_idx == -1:
                node.is_leaf = True
                return

            masks = [(float_eq(node.X[:, sep_feature_idx], value), value)
                     for value in np.unique(node.X[:, sep_feature_idx])]

            nodes = [tree.Node(
                node.X[mask],
                node.y[mask],
                node.depth + 1,
                sep_feature_idx,
                value,
                is_leaf=True,
                ID=next(id_gen)
            )
                for mask, value in masks]

            # remove leaves which contain less than min_samples_leaf samples
            node.children = list(filter(lambda nd: len(nd.X) >= self.min_samples_leaf, nodes))

            node.is_leaf = False
            node.sep_feature = sep_feature_idx

            if len(node.children) <= 1:
                node.is_leaf = True
                return

            for child in node.children:
                build(child)

# ----------------------------------------------------------------------------------

        split_dict = {
            'best': best_split,
            'random': random_split
        }

        # save the order of columns to reorder them in predict method
        if isinstance(X, pd.DataFrame):
            self.features = X.columns

        X = to_numpy(X)

        # TODO: get rid of double initialisation of binner objects ?

        self.__binners = [EqualLengthBinner(col) for col in X.T]

        X = np.apply_along_axis(
            lambda col: EqualLengthBinner(col).get_discrete(),
            axis=0,
            arr=X
        )

        y = to_numpy(y).reshape((-1, 1))

        num_classes = len(np.unique(y))

        use_count = [0] * len(X[0])

        self.__root = tree.Node(X, y, 0, 0, 0)
        build(self.__root)

        calc_probs(self.__root)

# ----------------------------------------------------------------------------------

    def print_tree(self):

        def dfs(node: tree.Node):
            print(node)
            for child in node.children:
                dfs(child)

        dfs(self.__root)

    def __predict_proba_row(self, row: np.ndarray) -> np.ndarray:
        node = self.__root

        while not node.is_leaf:
            # getting the discrete value of row[node.sep_feature]
            sep_feature_bin = self.__binners[node.sep_feature].get_bin(row[node.sep_feature])

            # getting the nearest child (by sep_feature)
            distances = np.array([abs(child.value - sep_feature_bin) for child in node.children])
            next_idx = np.argmin(distances)

            node = node.children[next_idx]

        return node.probs

    def predict_proba(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        X = self.reorder_columns(X)
        X = to_numpy(X)
        return np.array([self.__predict_proba_row(row) for row in X])

    def predict(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        probs_X = self.predict_proba(X)
        return np.array([np.argmax(probs) for probs in probs_X])
