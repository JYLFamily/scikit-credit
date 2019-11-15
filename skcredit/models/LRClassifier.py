# coding:utf-8

import gc
import logging
import numpy as np
import pandas as pd
from itertools import compress
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, c, keep_columns, random_state):
        self.c = c
        self.keep_columns = keep_columns
        self.random_state = random_state

        self.__feature_columns = None
        self.__model = None
        self.__coeff = None

    def fit(self, X, y):
        x = X.copy(deep=True)
        del X
        gc.collect()

        # perturb
        # from scipy.stats import variation
        # from sklearn.model_selection import StratifiedKFold
        #
        # self.__feature_columns = np.array([col for col in x.columns if col not in self.keep_columns])
        #
        # kfold = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
        # kfold_coeff = np.zeros((5, self.__feature_columns.shape[0]))
        #
        # for n_flod, (trn_idx, _) in enumerate(kfold.split(x, y)):
        #     model = LogisticRegression(
        #         C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        #     model.fit(x.iloc[trn_idx], y.iloc[trn_idx])
        #     kfold_coeff[n_flod, :] = model.coef_.reshape(-1, )
        #
        # self.__feature_columns = self.__feature_columns[variation(kfold_coeff, axis=0) <= 0.15]

        # variance inflation factor
        self.__feature_columns = np.array(
            [col for col in x.columns if col not in self.keep_columns])

        self.__model = LogisticRegression(
            C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
        self.__model.fit(x[self.__feature_columns], y)

        while np.any(self.__model.coef_.reshape(-1, ) < 0):
            vif_list = list()
            col_list = list()

            for col in list(compress(self.__feature_columns, self.__model.coef_.reshape(-1, ) < 0)):
                feature = np.setdiff1d(self.__feature_columns, np.array([col]))

                regressor = LinearRegression()
                regressor.fit(x.loc[:, feature], x.loc[:, col])

                vif_list.append(regressor.score(x.loc[:, feature], x.loc[:, col]))
                col_list.append(col)
            else:
                self.__feature_columns = np.setdiff1d(
                    self.__feature_columns, np.array([col_list[vif_list.index(max(vif_list))]]))
                logging.info(col_list[vif_list.index(max(vif_list))] + " remove !")

            self.__model = LogisticRegression(
                C=self.c, solver="lbfgs", max_iter=10000, random_state=self.random_state)
            self.__model.fit(x[self.__feature_columns], y)
            self.__coeff = self.__model.coef_.reshape(-1,)

        for col in self.__feature_columns:
            feature = np.setdiff1d(self.__feature_columns, np.array([col]))

            regressor = LinearRegression()
            regressor.fit(x.loc[:, feature], x.loc[:, col])

        return self

    def score(self, X, y, sample_weight=None):
        fpr, tpr, _ = roc_curve(y, self.predict_proba(X)[:, 1])

        return round(max(tpr - fpr), 5)

    def result(self):
        result = dict()
        result["column"] = self.__feature_columns.tolist()
        result["coefficient"] = np.round(self.__coeff, 5).tolist()

        return result

    def predic(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.__model.predict(
            x[self.keep_columns + self.__feature_columns.tolist()])

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.__model.predict_proba(
            x[self.keep_columns + self.__feature_columns.tolist()])

