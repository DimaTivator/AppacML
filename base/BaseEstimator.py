import inspect
from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    def _get_param_names(self):
        pass
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


    def get_params(self, deep=True):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

