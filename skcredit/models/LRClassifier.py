# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from itertools import compress
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
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

        # coefficient selection
        self.feature_columns_ = np.array(
            [col for col in x.columns if col not in self.keep_columns])

        self.model_ = LogisticRegression(
            C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        self.model_.fit(x[self.feature_columns_], y)
        self.coeff_ = self.model_.coef_.reshape(-1,)

        while np.any(self.coeff_ < 0):
            vif_list = list()
            col_list = list()

            for col in list(compress(self.feature_columns_, self.coeff_ < 0)):
                feature = np.setdiff1d(self.feature_columns_, np.array([col]))

                regressor = LinearRegression()
                regressor.fit(x.loc[:, feature], x.loc[:, col])

                col_list.append(col)
                vif_list.append(regressor.score(x.loc[:, feature], x.loc[:, col]))
            else:
                self.feature_columns_ = np.setdiff1d(
                    self.feature_columns_, np.array([col_list[vif_list.index(max(vif_list))]]))
                logging.info(col_list[vif_list.index(max(vif_list))] + " remove !")

            self.model_ = LogisticRegression(
                C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
            self.model_.fit(x[self.feature_columns_], y)
            self.coeff_ = self.model_.coef_.reshape(-1,)

        self.model_ = LogisticRegression(
            C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        self.model_.fit(x[self.feature_columns_], y)
        self.coeff_ = self.model_.coef_.reshape(-1,)

        return self

    def score(self, X, y=None, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X)[:, 1])

        return round(max(tpr - fpr), 5)

    def result(self):
        result = dict()
        result["column"] = self.feature_columns_.tolist()
        result["coefficient"] = np.round(self.coeff_, 5).tolist()

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

