class Node:
    def __init__(
            self,
            X,
            y,
            depth,
            sep_feature,
            sep_value,
            is_leaf=False
    ):
        self.is_leaf = is_leaf
        self.depth = depth
        self.children = []
        self.sep_feature = sep_feature
        self.sep_value = sep_value
        self.X = X
        self.y = y


if __name__ == "__main__":
    node = Node()
