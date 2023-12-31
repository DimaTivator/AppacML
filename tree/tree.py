class Node:
    def __init__(self, classes, is_leaf=False):
        self.is_leaf = is_leaf
        self.__children = []
        self.__sep_feature = 0
        self.__sep_values = []
        # leafs contain counts of each class in them
        self.classes_cnt = [0] * classes


if __name__ == "__main__":
    node = Node()

