from abc import ABC, abstractmethod

import numpy as np


class BaseProcessor(ABC):

    @staticmethod
    @abstractmethod
    def get_cols_names(obj) -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_cols(obj, indices) -> list:
        pass


class NumpyProcessor(BaseProcessor):

    @staticmethod
    def get_cols_names(obj) -> np.ndarray:
        if len(obj.shape) == 0:
            return np.array([])
        elif len(obj.shape) == 1:
            return np.arange(obj.shape[0])
        return np.arange(obj.shape[1])

    @staticmethod
    def get_cols(obj, indices) -> list:
        return obj[indices]


class PandasProcessor(BaseProcessor):

    @staticmethod
    def get_cols_names(obj) -> np.ndarray:
        return obj.columns

    @staticmethod
    def get_cols(obj, indices) -> list:
        return obj[indices]


class ListProcessor(BaseProcessor):

    @staticmethod
    def get_cols_names(obj) -> np.ndarray:
        if type(obj[0]) is list:
            return np.arange(obj[0])
        return np.arange(len(obj))

    @staticmethod
    def get_cols(obj, indices) -> list:
        return [obj[i] for i in indices]


class ProcessorFactory:
    @staticmethod
    def get_processor(obj) -> BaseProcessor:
        object_type = str(type(obj)).lower()

        if 'numpy' in object_type:
            return NumpyProcessor()
        elif 'pandas' in object_type:
            return PandasProcessor()
        elif 'list' in object_type:
            return ListProcessor()


class AppacDataFrame:
    pass
