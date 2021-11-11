# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from itertools import chain
from joblib import Parallel, delayed
from sklearn.base import clone, TransformerMixin
from sklearn.preprocessing   import   FunctionTransformer
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.pipeline import _fit_transform_one, _transform_one
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class Bucketer(_BaseComposition, TransformerMixin):
    def __init__(self, bucketers, remainder, nthread, verbose):
        super().__init__()
        self.bucketers = bucketers
        self.remainder = remainder
        self.nthread = nthread
        self.verbose = verbose

    @property
    def _bucketers(self):
        return [(name, buckter) for name, buckter, _ in self.bucketers]

    @_bucketers.setter
    def _bucketers(self,  value):
        self.bucketers = [
            (name, buckter, col)
            for ((name, buckter), (_, _, col)) in zip(value, self.bucketers)
        ]

    def get_params(self, deep=True):

        return self._get_params("_bucketers", deep=deep)

    def set_params(self,  **kwargs):

        self._set_params("_bucketers", **kwargs)
        return self

    def fit(self,    x, y=None):
        self.fit_transform(x, y=y)

        return self

    def transform(self, x):
        xs = self._fit_transform(x, None, _transform_one, fitted=True)

        return pd.concat((list(xs)), axis=1)

    def fit_transform( self, x, y=None,  **fit_params):
        self._validate_remainder(x)

        xs, bucketers = zip(*self._fit_transform(x, y, _fit_transform_one))

        self._update_fitted_bucketers(bucketers)

        return pd.concat((list(xs)), axis=1)

    def _fit_transform(self, x, y, func, fitted=False):
        return Parallel(n_jobs=self.nthread,     verbose=self.verbose)(
                delayed(func)(
                    transformer=clone(buckter) if not fitted else buckter,
                    X=x[column],
                    y=y,
                    weight=None)
                for name, buckter, column in self._iter(fitted=fitted))

    def _validate_remainder(self, x):
        remaining_column = set(x.columns.tolist()) - set(list(chain.from_iterable([column for _, _, column in self.buckters])))

        self._remainder = ('remainder', self.remainder, list(remaining_column))

    def _iter(self, fitted=False):
        if fitted:
            bucketers = self.bucketers_
        else:
            bucketers = self.bucketers

            if self._remainder[2] is not None:
                bucketers = chain(bucketers, [self._remainder])

        for name, buckter, column in bucketers:
            if  buckter  == "keep":
                buckter = FunctionTransformer(validate=False,
                    accept_sparse=False,
                    check_inverse=False)

            elif buckter == "drop":
                continue

            yield name, buckter, column

    def _update_fitted_bucketers(self, bucketers):
        fitted_bucketers =  iter(bucketers)
        bucketers_ = []

        for name, old, column in self._iter():
            if   old == "drop":
                buckter = "drop"
            elif old == "keep":
                next(fitted_bucketers)
                buckter = "keep"
            else:
                buckter = next(fitted_bucketers)
            bucketers_.append((name, buckter, column))

        self.bucketers_ = bucketers_


if __name__ == "__main__":
    # application_train = pd.read_csv("C:\\Users\\P1352\\Desktop\\application_train.csv",
    #                                 usecols=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "TARGET"]).head(20)
    # from skcredit.feature_bucketer.SplitNumND import SplitNumND
    # buckter = Bucketer(
    #
    #     [("EXT_SOURCE_1", SplitNumND(), ["EXT_SOURCE_1"]),
    #      ("EXT_SOURCE_1@EXT_SOURCE_2", SplitNumND(), ["EXT_SOURCE_1", "EXT_SOURCE_2"])],
    #     remainder='keep',
    #     nthread=1,
    #     verbose=1
    # )
    # print(buckter.fit_transform(application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]],
    #                       application_train["TARGET"]).head())
    pass
