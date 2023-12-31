from utils import *


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
            np.array([-INF])
        ),
            axis=0
        )

    def get_bin(self, value):
        return np.argmax([self.__starts[i] <= value < self.__ends[i] for i in range(self.num_bins + 1)])

