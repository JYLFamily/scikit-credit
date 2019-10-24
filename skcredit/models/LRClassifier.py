# coding: utf-8

import gc
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class LRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, param, keep_columns):
        self.__param = param
        self.__keep_columns = keep_columns

        self.__feature_columns = None
        self.__feature_support = None
        self.__model = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        # discretize check
        self.__feature_columns = np.array([col for col in x.columns if col not in self.__keep_columns])
        self.__feature_support = np.zeros_like(self.__feature_columns, dtype=bool)

        beta_0 = np.log(y.sum() / (len(y) - y.sum()))
        beta_1 = 1.

        for idx, col in enumerate(self.__feature_columns):
            logit_mod = sm.Logit(y, sm.add_constant(x[[col]], prepend=False))
            logit_res = logit_mod.fit()

            if (np.abs(logit_res.params["const"] - beta_0) < 0.0001 and
                    np.abs(logit_res.params[col] - beta_1) < 0.0001):
                self.__feature_support[idx] = True

        # fit multivariate model
        self.__feature_columns = self.__feature_columns[self.__feature_support]

        return self

    def predict(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.__model.predic(x[self.__feature_columns])

    def predict_proba(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        return self.__model.predict_proba(x[self.__feature_columns])


if __name__ == "__main__":
    train = pd.read_csv("E:\\BDD+RJK+JG\\0.8\\TRA_woe.csv", encoding="gbk")
    oot = pd.read_csv("E:\\BDD+RJK+JG\\0.8\\oot_woe.csv", encoding="gbk")

    train = train.drop(["姓名", "身份证号", "手机号", "申请时间"], axis=1)
    train_feature, train_label = train.drop(["target"], axis=1).copy(deep=True), train["target"].copy(deep=True)

    oot = oot.drop(["姓名", "身份证号", "手机号", "申请时间"], axis=1)
    oot_feature, oot_label = oot.drop(["target"], axis=1).copy(deep=True), oot["target"].copy(deep=True)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    pca.fit(train_feature)
    train_feature = pca.transform(train_feature)
    # oot_feature = pca.transform(oot_feature)

    logit_mod = sm.Logit(train_label, sm.add_constant(train_feature, prepend=False))
    logit_res = logit_mod.fit()
    print(logit_res.summary())

    # lrc = LRClassifier(param=None, keep_columns=[])
    # lrc.fit(train_feature, train_label)

