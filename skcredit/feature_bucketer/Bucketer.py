# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from itertools import chain
from joblib import Parallel, delayed
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import _fit_one, _transform_one
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.metaestimators import _BaseComposition
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
                for name, buckter, column  in self._iter(False))

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
                for name, buckter, column   in self._iter(True))

        return pd.concat((list(xs)), axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)

    def _iter(self, fitted):
        if fitted:
            bucketers = self.bucketers_
        else:
            bucketers = self.bucketers
            bucketers.extend(self.remainder)

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

        for name, old, column in self._iter(False):
            if   old == "keep":
                next(fitted_bucketers)
                buckter = "keep"
            elif old == "drop":
                buckter = "drop"
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
    from sklearn.preprocessing import Normalizer