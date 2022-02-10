
# Hurdle

![tests status](https://github.com/prio-data/cc_backend_lib/actions/workflows/test.yml/badge.svg)

This package contains an implementation of Hurdle Regression, based in part on
[Geoff Ruddocks implementation](https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/)
and Håvard Hegres 2022 adaption of his implementation.

## Usage

```
from hurdle import HurdleEstimator
from sklearn import linear_model

est = HurdleEstimator(linear_model.LogisticRegression(), linear_model.LinearModel())

est.fit(...)
```
