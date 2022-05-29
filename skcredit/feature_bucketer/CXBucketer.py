# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import   clone,    TransformerMixin
from sklearn.pipeline import _fit_one, _transform_one
from sklearn.utils.metaestimators import _BaseComposition
np.random.seed(7)
pd.set_option("styler.render.max_rows"   , 500)
pd.set_option("styler.render.max_columns", 500)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class CXBucketer(_BaseComposition, TransformerMixin):
    def __init__(self,  bucketers, nthread, verbose):
        super().__init__()
        self.bucketers = bucketers
        self.nthread = nthread
        self.verbose = verbose

    @property
    def _bucketers(self):
        return [(name, buckter) for name, buckter, _ in self.bucketers]

    @_bucketers.setter
    def _bucketers(self, value):
        self.bucketers = [
            (name, buckter, col)
            for ((name, buckter), (_, _, col)) in zip(value, self.bucketers)
        ]

    def get_params(self, deep=True):

        return self._get_params("_bucketers", deep=deep)

    def set_params(self,  **kwargs):
        self._set_params("_bucketers", **kwargs)

        return self

    def fit(self,    X, y):
        buckters = \
            Parallel(n_jobs=self.nthread, verbose=self.verbose)(
                delayed(_fit_one)(
                    transformer=clone(buckter),
                    X=X[column],
                    y=y,
                    weight=None)
                for name, buckter, column in self._iter())

        self._update_fitted_bucketers(buckters)

        return self

    def transform(self, X):
        xs = \
            Parallel(n_jobs=self.nthread, verbose=self.verbose)(
                delayed(_transform_one)(
                    transformer=buckter,
                    X=X[column],
                    y=None,
                    weight=None)
                for name, buckter, column in self._iter())

        return pd.concat((list(xs)), axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)

    def _iter(self):
        for name, buckter, column in self.bucketers:
            yield name, buckter, column

    def _update_fitted_bucketers(self, bucketers):
        fitted_bucketers  = iter(bucketers)
        bucketers_ = list()

        for name, _, column in self.bucketers:
            bucketers_.append((name, next(fitted_bucketers), column))

        self.bucketers = bucketers_

