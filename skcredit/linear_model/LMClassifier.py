# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def inclusion(X, y, feature_columns, feature_subsets):
    x = X.copy(deep=True)
    del X
    gc.collect()

    feature_remains = feature_columns - feature_subsets
    feature_pvalues = dict()

    for col in feature_remains:
        logit_mod = sm.GLM(y, sm.add_constant(x[list(feature_subsets | {col})]), family=sm.families.Binomial())
        logit_res = logit_mod.fit()
        feature_pvalues[col] = logit_res.pvalues[col]

    return feature_pvalues


def exclusion(X, y, feature_subsets):
    x = X.copy(deep=True)
    del X
    gc.collect()

    logit_mod = sm.GLM(y, sm.add_constant(x[list(feature_subsets)]), family=sm.families.Binomial())
    logit_res = logit_mod.fit()

    return logit_res.pvalues.drop("const").to_dict()


def stepwises(X, y, feature_columns, feature_subsets, lrmodel_pvalues):
    x = X.copy(deep=True)
    del X
    gc.collect()

    feature_pvalues = inclusion(x, y, feature_columns, feature_subsets)
    while min(feature_pvalues.values()) < lrmodel_pvalues:
        feature_subsets = feature_subsets | {min(feature_pvalues, key=feature_pvalues.get)}

        feature_pvalues = exclusion(x, y, feature_subsets)
        while max(feature_pvalues.values()) > lrmodel_pvalues:
            feature_subsets = feature_subsets - {max(feature_pvalues, key=feature_pvalues.get)}
            feature_pvalues = exclusion(x, y, feature_subsets)

        feature_pvalues = inclusion(x, y, feature_columns, feature_subsets)

    logit_mod = sm.GLM(y, sm.add_constant(x[list(feature_subsets)]), family=sm.families.Binomial())
    logit_res = logit_mod.fit()

    return logit_res, feature_subsets


class LMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keep_columns, PDO, BASE, ODDS):
        self.keep_columns = keep_columns

        self.feature_columns_ = None
        self.feature_subsets_ = None
        self.model_ = None
        self.coeff_ = None

        self.b_ = PDO / np.log(2)
        self.a_ = BASE - self.b_ * np.log(ODDS)

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        lrmodel_pvalues = 0.001
        self.feature_columns_ = set([col for col in x.columns if col not in self.keep_columns])
        self.feature_subsets_ = set()

        self.model_, self.feature_subsets_ = stepwises(
            x, y,
            self.feature_columns_,
            self.feature_subsets_,
            lrmodel_pvalues
        )
        self.coeff_ = self.model_.params.drop("const")

        while np.any(self.coeff_ < 0):
            lrmodel_pvalues /= 10
            self.feature_subsets_ = set()

            self.model_, self.feature_subsets_ = stepwises(
                x, y,
                self.feature_columns_,
                self.feature_subsets_,
                lrmodel_pvalues
            )
            self.coeff_ = self.model_.params.drop("const")

        return self

    def score(self, X, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X))

        return round(max(tpr - fpr), 5)

    def result(self):

        return self.model_.summary()

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.model_.predict(
            sm.add_constant(x[list(self.feature_subsets_)]))

    def predict_score(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        y = np.log(self.predict_proba(x) / (1 - self.predict_proba(x)))

        return np.round(self.a_ + self.b_ * (- y))

