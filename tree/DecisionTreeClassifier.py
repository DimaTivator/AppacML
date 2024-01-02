from utils import *
from binning import EqualLengthBinner, is_discrete
import criterion as crit
import tree.tree as tree


class DecisionTreeClassifier:
    """
    Attributes
    -------------
    criterion: str
        The function to measure the quality of split
        {'gini', 'entropy'}

        ...

    Methods
    -------------

    ...

    """

    def __init__(
            self,
            criterion='gini',
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

        self.criterion_dict = {
            'gini': crit.gini,
            'entropy': crit.entropy
        }

    def __is_terminal(self, node: tree.Node):
        return (
                node.depth == self.max_depth or
                len(np.unique(node.y)) == 1
        )

    def fit(self, X, y):

        id_gen = int_gen()

# ----------------------------------------------------------------------------------

        def build(node: tree.Node):
            if self.__is_terminal(node):
                node.is_leaf = True
                return

            min_score = INF
            sep_feature_idx = 0

            # best split algorithm
            for i in range(len(node.X[0])):
                col = node.X[:, i]
                s_split = [node.y[float_eq(node.X[:, i], value)] for value in np.unique(col)]

                split_score = sum([self.criterion_dict[self.criterion](s_split[i])
                                   for i in range(len(s_split))])

                if split_score < min_score:
                    min_score = split_score
                    sep_feature_idx = i

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

            node.children = nodes
            node.is_leaf = False
            node.sep_feature = sep_feature_idx

            if len(node.children) <= 1:
                return

            for child in node.children:
                build(child)

# ----------------------------------------------------------------------------------

        X = np.apply_along_axis(
            lambda col: EqualLengthBinner(col).get_discrete(),
            axis=0,
            arr=to_numpy(X)
        )
        y = to_numpy(y)

        # use_count[i] -- the number of splits by i-th feature
        # TODO
        # use_count = [0] * len(X[0])

        root = tree.Node(X, y, 0, 0, 0)
        build(root)
