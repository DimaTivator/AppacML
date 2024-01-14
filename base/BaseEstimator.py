import inspect
from abc import ABC, abstractmethod


class BaseEstimator(ABC):

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

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

