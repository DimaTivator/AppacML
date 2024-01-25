import inspect
import metrics
from abc import ABC, abstractmethod
from typing import Callable
from types import ModuleType
import numpy as np
import pandas as pd


def get_functions_by_substring(package: ModuleType, substring: str):
    function_names = [name for name in dir(package) if callable(getattr(package, name))]
    substring_functions = [func_name for func_name in function_names if substring in func_name.lower()]
    substring_functions_dict = {func_name: getattr(metrics, func_name) for func_name in substring_functions}
    return substring_functions_dict


class BaseEstimator(ABC):

    # decorator to make all children implement __init__ method
    @abstractmethod
    def __init__(self):
        self.features = []

    @classmethod
    def _get_init_attributes_names(cls):

        """
        Checking if the given class provides __init__ method implementation
        In this case we'll take the attributes from its signature
        Otherwise we'll look fot the attribute names in cls.__dict__
        """

        if hasattr(cls, '__init__'):
            init_signature = inspect.signature(cls.__init__)
            return [param for param in init_signature.parameters.keys() if param != 'self']

        return [attr for attr in cls.__dict__ if not callable(getattr(cls, attr)) and not attr.startswith("__")]

    def get_params(self, **kwargs):

        """
        Gets estimator's parameters

        ! It's assumed that all parameters are defined in __init__ method signature
        """

        attribute_names = self._get_init_attributes_names()
        return {name: getattr(self, name) for name in attribute_names}

    def set_params(self, new_params):

        """
        Set the values of the estimator's parameters based on the input dictionary.

        Parameters:
            new_params (dict): Dictionary containing new parameter values.

        Raises:
            ValueError: If a parameter key does not correspond to an existing attribute in the class.
        """

        for key, value in new_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Estimator has no attribute named {key}')

    def reorder_columns(self, df):
        if not isinstance(df, pd.DataFrame):
            return df

        if set(df.columns) != set(self.features):
            raise ValueError('The estimator was fitted on other features')

        reordered_df = df[self.features].copy()
        return reordered_df

    @abstractmethod
    def fit(self, X, y):
        pass


class BaseSearch(BaseEstimator, ABC):

    @abstractmethod
    def __init__(
            self,
            estimator: BaseEstimator,
            scoring: str | Callable[[np.ndarray, np.ndarray], float] = None,
    ):
        super().__init__()

        self.estimator = estimator
        self.scoring = scoring

        self.scoring_functions = get_functions_by_substring(metrics, 'score')

        self._best_estimator = None
        self._best_score = None
        self._best_params = None

    @property
    def best_estimator(self):
        return self._best_estimator

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_params(self):
        return self._best_params

    def score(self, X, y) -> float:
        # TODO implement
        pass

    @abstractmethod
    def fit(self, X, y):
        pass
