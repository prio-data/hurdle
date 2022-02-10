from typing import Optional

from sklearn import ensemble, linear_model

try:
    from xgboost import XGBRegressor
    from xgboost import XGBClassifier
    from xgboost import XGBRFRegressor, XGBRFClassifier

    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    raise ImportError("the menu_hurdle_estimator requires xboost and lightgbm. Please install before running")

from . import hurdle

class MenuHurdleEstimator(hurdle.HurdleEstimator):
    def __init__(self,
                 clf_name: str = 'logistic',
                 reg_name: str = 'linear',
                 clf_params: Optional[dict] = None,
                 reg_params: Optional[dict] = None,
                 n_jobs: int = 4):

        self._n_jobs = n_jobs

        threshold_estimator = self._resolve_estimator(clf_name)
        threshold_estimator.set_params(**clf_params)

        regression_estimator = self._resolve_estimator(reg_name)
        regression_estimator.set_params(**reg_params)

        super().__init__(threshold_estimator, regression_estimator)

    def _resolve_estimator(self, func_name: str):
        """ Lookup table for supported estimators.
        This is necessary because sklearn estimator default arguments
        must pass equality test, and instantiated sub-estimators are not equal. """

        funcs = {'linear':         linear_model.LinearRegression(),
                 'logistic':       linear_model.LogisticRegression(solver = 'liblinear'),
                 'LGBMRegressor':  LGBMRegressor(n_estimators = 100),
                 'LGBMClassifier': LGBMClassifier(n_estimators = 100),
                 'RFRegressor':    XGBRFRegressor(n_estimators = 300, n_jobs = self._n_jobs),
                 'RFClassifier':   XGBRFClassifier(n_estimators = 300, n_jobs = self._n_jobs),
                 'GBMRegressor':   ensemble.GradientBoostingRegressor(n_estimators = 200),
                 'GBMClassifier':  ensemble.GradientBoostingClassifier(n_estimators = 200),
                 'XGBRegressor':   XGBRegressor(n_estimators = 200, tree_method = 'hist', n_jobs = self._n_jobs),
                 'XGBClassifier':  XGBClassifier(n_estimators = 200, tree_method = 'hist', n_jobs = self._n_jobs, use_label_encoder = False),
                 'HGBRegressor':   ensemble.HistGradientBoostingRegressor(max_iter = 200),
                 'HGBClassifier':  ensemble.HistGradientBoostingClassifier(max_iter = 200),
                }

        return funcs[func_name]

