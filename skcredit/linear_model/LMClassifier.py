# coding:utf-8

import gc
import logging
import tempfile
import numpy as np
import pandas as pd
from joblib import Memory
from itertools import count
from sklearn.metrics import roc_curve
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def first_feature_in_group(labels):
    d = dict()

    for idx, val in enumerate(labels):
        if val not in d.keys():
            d[val] = idx

    return np.array(list(d.values()))


class LMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C, PDO, BASE, ODDS, keep_columns, random_state):
        self.C = C
        self.keep_columns = keep_columns
        self.random_state = random_state

        self.feature_columns_ = None
        self.model_ = None
        self.coeff_ = None

        self.b_ = PDO / np.log(2)
        self.a_ = BASE - self.b_ * np.log(ODDS)

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.feature_columns_ = np.array(
            [col for col in x.columns if col not in self.keep_columns])

        # # iter counter
        # counter = count(15, -1) if len(self.feature_columns_) > 15 else count(len(self.feature_columns_), -1)
        #
        # # feature agglomeration
        # ward = FeatureAgglomeration(
        #     n_clusters=next(counter), memory=Memory(location=tempfile.mkdtemp(), verbose=0))
        # ward.fit(x[self.feature_columns_])
        # self.feature_columns_ = self.feature_columns_[first_feature_in_group(ward.labels_)]
        #
        # self.model_ = LogisticRegression(
        #     C=self.C, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        # self.model_.fit(x[self.feature_columns_], y)
        # self.coeff_ = self.model_.coef_.reshape(-1,)
        #
        # while np.any(self.coeff_ < 0):
        #     ward = FeatureAgglomeration(
        #         n_clusters=next(counter), memory=Memory(location=tempfile.mkdtemp(), verbose=0))
        #     ward.fit(x[self.feature_columns_])
        #     self.feature_columns_ = self.feature_columns_[first_feature_in_group(ward.labels_)]
        #
        #     self.model_ = LogisticRegression(
        #         C=self.C, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        #     self.model_.fit(x[self.feature_columns_], y)
        #     self.coeff_ = self.model_.coef_.reshape(-1,)
        # self.model_ = LogisticRegression(
        #     C=self.C, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        self.model_ = LogisticRegression(
            C=self.C, max_iter=10000, random_state=self.random_state)
        self.model_.fit(x[self.feature_columns_], y)
        self.coeff_ = self.model_.coef_.reshape(-1,)

        return self

    def score(self, X, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X)[:, 1])

        return round(max(tpr - fpr), 5)

    def result(self):
        result = dict()

        column = self.feature_columns_.tolist()
        column.append("intercept")
        coeffs = self.coeff_.tolist()
        coeffs.append(self.model_.intercept_[0])

        for k, v in zip(column, coeffs):
            result[k] = v

        return result

    def predic(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.model_.predict(
            x[self.feature_columns_.tolist()])

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.model_.predict_proba(
            x[self.feature_columns_.tolist()])

    def predict_score(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        y = np.log(self.predict_proba(x)[:, 1] / self.predict_proba(x)[:, 0])

        return np.round(self.a_ + self.b_ * (- y))

