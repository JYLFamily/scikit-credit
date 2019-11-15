# coding:utf-8

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.DiscreteUtil import merge_num_table, merge_cat_table
from skcredit.feature_discretization.DiscreteUtil import replace_num_woe, replace_cat_woe
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
plt.style.use("ggplot")


class Discrete(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            keep_columns, cat_columns, num_columns,
            merge_bin, information_value_threshold=0.1):
        self.__keep_columns = keep_columns
        self.__cat_columns = cat_columns
        self.__num_columns = num_columns

        self.__merge_bin = merge_bin
        self.__information_value_threshold = information_value_threshold

        self.cat_table_ = dict()
        self.num_table_ = dict()

        self.cat_columns_ = None
        self.num_columns_ = None

        self.information_values_ = OrderedDict()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        with Pool() as pool:
            if len(self.__cat_columns) != 0:
                self.cat_table_ = dict(zip(self.__cat_columns, pool.starmap(
                    merge_cat_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.__merge_bin) for col in
                        self.__cat_columns])))
        self.cat_table_ = {
            col: val for col, val in self.cat_table_.items() if val["IV"].sum() > self.__information_value_threshold}

        with Pool() as pool:
            if len(self.__num_columns) != 0:
                self.num_table_ = dict(zip(self.__num_columns, pool.starmap(
                    merge_num_table,
                    [(pd.concat([x[[col]], y.to_frame("target")], axis=1), col, self.__merge_bin) for col in
                        self.__num_columns])))
        self.num_table_ = {
            col: val for col, val in self.num_table_.items() if val["IV"].sum() > self.__information_value_threshold}

        self.information_values_.update({col: val["IV"].sum() for col, val in self.cat_table_.items()})
        self.information_values_.update({col: val["IV"].sum() for col, val in self.num_table_.items()})
        self.information_values_ = OrderedDict(
            sorted(self.information_values_.items(), key=lambda t: t[1], reverse=True))

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.num_columns_ = list(self.num_table_.keys())
        self.cat_columns_ = list(self.cat_table_.keys())

        if len(self.cat_columns_) != 0:
            x = x.drop(list(set(self.__cat_columns).difference(self.cat_columns_)), axis=1)

            for col in self.cat_table_.keys():
                woe = self.cat_table_[col]["WoE"].tolist()
                categories = self.cat_table_[col][col].tolist()
                x[col] = x[col].apply(lambda element: replace_cat_woe(element, categories, woe))

        if len(self.num_columns_) != 0:
            x = x.drop(list(set(self.__num_columns).difference(self.num_columns_)), axis=1)

            for col in self.num_table_.keys():
                woe = self.num_table_[col]["WoE"].tolist()
                upper = self.num_table_[col]["Upper"].tolist()
                x[col] = x[col].apply(lambda element: replace_num_woe(element, upper, woe))

        return x[self.__keep_columns + list(self.information_values_.keys())]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)