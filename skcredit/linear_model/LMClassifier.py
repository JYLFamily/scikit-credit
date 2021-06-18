# coding:utf-8

import gc
import logging
import numpy  as np
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
        logit_mod = sm.GLM(
            y, sm.add_constant(x[list(feature_subsets | {col})], has_constant="add"), family=sm.families.Binomial())
        logit_res = logit_mod.fit()
        feature_pvalues[col] = logit_res.pvalues[col]

    return feature_pvalues


def exclusion(X, y, feature_subsets):
    x = X.copy(deep=True)
    del X
    gc.collect()

    logit_mod = sm.GLM(
        y, sm.add_constant(x[list(feature_subsets)], has_constant="add"), family=sm.families.Binomial())
    logit_res = logit_mod.fit()

    return logit_res.pvalues.drop("const").to_dict()


def stepwises(X, y, feature_columns, feature_subsets, lrmodel_pvalues):
    x = X.copy(deep=True)
    del X
    gc.collect()

    feature_pvalues = inclusion(x, y, feature_columns, feature_subsets)
    inclusion_flags = min(feature_pvalues.values()) < lrmodel_pvalues

    while inclusion_flags:
        feature_subsets = feature_subsets | {min(feature_pvalues, key=feature_pvalues.get)}
        feature_pvalues = exclusion(x, y, feature_subsets)
        exclusion_flags = max(feature_pvalues.values()) > lrmodel_pvalues

        while exclusion_flags:
            feature_subsets = feature_subsets - {max(feature_pvalues, key=feature_pvalues.get)}
            feature_pvalues = exclusion(x, y, feature_subsets)
            exclusion_flags = max(feature_pvalues.values()) > lrmodel_pvalues

        feature_pvalues = inclusion(x, y, feature_columns, feature_subsets)
        inclusion_flags = bool(feature_pvalues) and (min(feature_pvalues.values()) < lrmodel_pvalues)

    logit_mod = sm.GLM(y, sm.add_constant(x[list(feature_subsets)]), family=sm.families.Binomial())
    logit_res = logit_mod.fit()

    return logit_res


class LMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tim_columns, PDO, BASE, ODDS):
        self.tim_columns = tim_columns

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

        lrmodel_pvalues = 0.01
        self.feature_columns_ = set([col for col in x.columns if col not in self.tim_columns])
        self.feature_subsets_ = set()

        self.model_ = stepwises(
            x, y,
            self.feature_columns_,
            self.feature_subsets_,
            lrmodel_pvalues
        )

        while np.any(self.model_.params.drop("const") < 0):
            lrmodel_pvalues /= 10

            self.model_ = stepwises(
                x, y,
                self.feature_columns_,
                self.feature_subsets_,
                lrmodel_pvalues
            )

        self.coeff_ = self.model_.params
        self.feature_subsets_ = self.coeff_.drop("const").index.tolist()

        return self

    def model(self):
        return self.model_.summary()

    def score(self, X, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X)["proba_positive"])

        return round(max(tpr - fpr), 5)

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        proba_positive = self.model_.predict(sm.add_constant(x[self.feature_subsets_])).to_frame("proba_positive")
        proba_negative = (1 - proba_positive.squeeze()).to_frame("proba_negative")

        return pd.concat([proba_negative, proba_positive], axis=1)

    def predict_score(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        y = np.log(self.predict_proba(x)["proba_positive"] / (1 - self.predict_proba(x)["proba_positive"]))

        return self.a_ + self.b_ * (- y)

