
import numpy as np
from sklearn import base
from sklearn.utils import validation
from sklearn.exceptions import NotFittedError

class HurdleEstimator(base.BaseEstimator):

    def __init__(self,
            threshold_estimator: base.BaseEstimator,
            regression_estimator: base.BaseEstimator):

        self.threshold_estimator = threshold_estimator
        self.regression_estimator = regression_estimator

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):

        X, y = validation.check_X_y(X, y, dtype=None,
                         accept_sparse=False,
                         accept_large_sparse=False,
                         force_all_finite='allow-nan')

        if X.shape[1] < 2:
            raise ValueError('Cannot fit model when n_features = 1')

        self.threshold_estimator.fit(X, y > 0)
        self.regression_estimator.fit(X[y > 0], y[y > 0])

        return self

    def _predict(self, X: np.ndarray, binary: bool = False):
        X = validation.check_array(X, accept_sparse=False, accept_large_sparse=False)
        validation.check_is_fitted(self)

        threshold_predict_method = "predict_proba" if not binary else "predict"
        threshold_predict = getattr(self.threshold_estimator, threshold_predict_method)
        threshold = threshold_predict(X)[:, 1]
        regression = self.regression_estimator.predict(X)
        return threshold * regression

    def predict(self, X: np.ndarray):
        """ Predict combined response using probabilistic classification outcome """
        return self._predict(X)

    def predict_bck(self, X: np.ndarray):
        """ Predict combined response using binary classification outcome """
        return self._predict(X, binary = True)

    def __sklearn_is_fitted__(self):
        return (
            self._is_fitted(self.regression_estimator) &
            self._is_fitted(self.threshold_estimator))

    @staticmethod
    def _is_fitted(estimator: base.BaseEstimator) -> bool:
        try:
            validation.check_is_fitted(estimator)
            return True
        except NotFittedError:
            return False
