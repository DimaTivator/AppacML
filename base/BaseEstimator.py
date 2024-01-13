from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    def _get_param_names(self):
        pass

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

