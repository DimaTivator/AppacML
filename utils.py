import numpy as np
import pandas as pd
import math

INF = float('inf')
EPS = 1e-7


def to_numpy(obj) -> np.array:
    if type(obj) is pd.DataFrame:
        return obj.values

    if type(obj) is pd.Series:
        return np.reshape(obj.values, (len(obj.values), 1))

    if type(obj) is list:
        return np.array(obj)


def float_eq(a: float, b: float) -> bool:
    return abs(a - b) < EPS


def int_gen():
    i = 1
    while True:
        yield i
        i += 1

