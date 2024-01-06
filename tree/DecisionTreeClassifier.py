from utils import *
from binning import EqualLengthBinner
import criterion as crit
import tree.tree as tree


class DecisionTreeClassifier:
    """
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
        X, y: pandas.DataFrame | pandas.Series | numpy.array | list
        Builds a decision tree. Building algorithm -- ID3
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
            random_state=None,
    ):

        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

        self.criterion_dict = {
            'gini': crit.gini,
            'entropy': crit.entropy
        }

        self.random_generator = np.random.RandomState(self.random_state)
        self.__root = None

    def __is_terminal(self, node: tree.Node):
        return (
                node.depth == self.max_depth or
                len(np.unique(node.y)) == 1
        )

    def fit(self, X, y):

        id_gen = int_gen()

# ----------------------------------------------------------------------------------

        def get_split(node: tree.Node, idx: int) -> list:
            col = node.X[:, idx]
            return [node.y[float_eq(node.X[:, idx], value)] for value in np.unique(col)]

        def calc_proba(node: tree.Node) -> list:
            if not node.is_leaf:
                return []

            n = len(node.y)  # the total number of rows
            classes = np.unique(node.y)
            probs = [0] * len(classes)

            for i, c in enumerate(classes):
                probs[i] = np.count_nonzero(node.y == c) / n

            return probs

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

            if self.__is_terminal(node) or sep_feature_idx == -1:
                node.is_leaf = True
                node.probs = calc_proba(node)
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
                return

            for child in node.children:
                build(child)

# ----------------------------------------------------------------------------------

        split_dict = {
            'best': best_split,
            'random': random_split
        }

        X = np.apply_along_axis(
            lambda col: EqualLengthBinner(col).get_discrete(),
            axis=0,
            arr=to_numpy(X)
        )
        y = to_numpy(y)

        use_count = [0] * len(X[0])

        self.__root = tree.Node(X, y, 0, 0, 0)
        build(self.__root)

# ----------------------------------------------------------------------------------

    def print_tree(self):

        def dfs(node: tree.Node):
            print(node)
            for child in node.children:
                dfs(child)

        dfs(self.root)
        dfs(self.__root)
