from utils import *


def is_discrete(array):
    # we consider a feature to be discrete if it has less than 20 distinct values
    return len(np.unique(array)) <= 20


class EqualLengthBinner:

    def __init__(self, array):
        self.array = array
        self.__create_bins()

    def __create_bins(self):
        array_min = np.min(self.array)
        array_max = np.max(self.array)
        self.num_bins = math.ceil(np.log2(len(self.array)))

        self.__starts = np.concatenate((
            np.array([-INF]),
            np.linspace(array_min, array_max, self.num_bins)
        ),
            axis=0
        )

        self.__ends = np.concatenate((
            np.linspace(array_min, array_max, self.num_bins),
            np.array([INF])
        ),
            axis=0
        )

    def __get_bin(self, value):
        return np.argmax([self.__starts[i] <= value < self.__ends[i] for i in range(self.num_bins + 1)])

    def get_discrete(self):
        if is_discrete(self.array):
            return self.array

        return np.array([self.__get_bin(value) for value in self.array])

