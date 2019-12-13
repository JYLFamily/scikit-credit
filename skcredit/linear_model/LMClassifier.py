# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from itertools import count
from skcredit.linear_model import ks_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from mlxtend.feature_selection import SequentialFeatureSelector
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


class LRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, c, keep_columns, random_state):
        self.c = c
        self.keep_columns = keep_columns
        self.random_state = random_state

        self.feature_columns_ = None
        self.model_ = None
        self.coeff_ = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        # iter counter
        counter = count(25, -1)

        # embed feature selection
        self.feature_columns_ = np.array(
            [col for col in x.columns if col not in self.keep_columns])

        select = SequentialFeatureSelector(
            estimator=LogisticRegression(
                C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state),
            k_features=next(counter),
            forward=True, floating=True,
            scoring=ks_score,
            cv=None, n_jobs=-1
        )
        select.fit(x[self.feature_columns_], y)

        self.model_ = LogisticRegression(
            C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        self.model_.fit(
            x[np.array(select.k_feature_names_)], y)
        self.coeff_ = self.model_.coef_.reshape(-1,)
        del select

        while np.any(self.coeff_ < 0):
            select = SequentialFeatureSelector(
                estimator=LogisticRegression(
                    C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state),
                k_features=next(counter),
                forward=True, floating=True,
                scoring=ks_score,
                cv=None, n_jobs=-1
            )
            select.fit(x[self.feature_columns_], y)

            self.model_ = LogisticRegression(
                C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
            self.model_.fit(
                x[np.array(select.k_feature_names_)], y)
            self.coeff_ = self.model_.coef_.reshape(-1,)
        else:
            self.feature_columns_ = np.array(
                select.k_feature_names_)

        return self

    def score(self, X, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X)[:, 1])

        return round(max(tpr - fpr), 5)

    def result(self):
        result = dict()

        result["column"] = self.feature_columns_.tolist()
        result["column"].append("intercept")

        result["coefficient"] = np.round(self.coeff_, 5).tolist()
        result["coefficient"].append(self.model_.intercept_[0])

        return result

    def predic(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.model_.predict(
            x[self.keep_columns + self.feature_columns_.tolist()])

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.model_.predict_proba(
            x[self.keep_columns + self.feature_columns_.tolist()])

