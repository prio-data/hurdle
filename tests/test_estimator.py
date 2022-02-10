
import unittest
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from hurdle import HurdleEstimator

class TestHurdleEstimator(unittest.TestCase):
    def setUp(self):

        X      = np.random.rand(1000, 4)
        y      = X.prod(axis = 1) + ((np.random.rand(1000)-.5) * .1)

        thresh = np.random.choice([0,1],1000,p = [.3,.7])
        flip   = np.random.choice([0,1],1000, p = [.2, .8])

        self.y = (y * thresh)
        self.X = np.concatenate([X, (thresh * flip)[:, np.newaxis]], axis = 1)

    def test_est(self):
        """
        Verifies that a hurdle estimator does better at modelling the
        bi-process data defined in the setUp method, than a regular
        LinearRegression does (it has a lower MSE).
        """

        est = HurdleEstimator(
                linear_model.LogisticRegression(),
                linear_model.LinearRegression())

        just_reg = linear_model.LinearRegression()

        est.fit(self.X, self.y)
        just_reg.fit(self.X, self.y)

        hurdle_mse = mean_squared_error(self.y, est.predict(self.X))
        just_reg_mse = mean_squared_error(self.y, just_reg.predict(self.X))

        self.assertGreater(just_reg_mse, hurdle_mse)
