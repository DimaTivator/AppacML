import numpy as np
import math

import pandas as pd

INF = float('inf')


def to_numpy(obj):
    if type(obj) is pd.DataFrame:
        return obj.values

    if type(obj) is pd.Series:
        return np.reshape(obj.values, (len(obj.values), 1))

    if type(obj) is list:
        return np.array(obj)


