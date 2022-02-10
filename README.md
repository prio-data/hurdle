
# Hurdle

![tests status](https://github.com/prio-data/cc_backend_lib/actions/workflows/test.yml/badge.svg)

This package contains an implementation of Hurdle Regression, based in part on
[Geoff Ruddocks implementation](https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/)
and Håvard Hegres 2022 adaption of his implementation.

## Installation

```
pip install hurdle
```

To use the `hurdle.menu_hurdle_estimator.MenuHurdleEstimator` module you also
need to install the `xgboost` and `lightgbm` packages.

## Usage

```
from hurdle import HurdleEstimator
from sklearn import linear_model

est = HurdleEstimator(linear_model.LogisticRegression(), linear_model.LinearModel())

est.fit(...)
```
