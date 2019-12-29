# coding: utf-8

import gc
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.DiscreteImple import replace_cat_woe, replace_num_woe


class BaseDiscrete(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns):
        self.keep_columns = keep_columns
        self.cat_columns_ = None
        self.num_columns_ = None

        self.cat_table_ = dict()
        self.num_table_ = dict()

        self.information_values_ = OrderedDict()

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        x = x[self.keep_columns + list(self.information_values_.keys())]

        if self.cat_table_.keys():
            for col in self.cat_table_.keys():
                woe = self.cat_table_[col]["WoE"].tolist()
                group_list = self.cat_table_[col][col].tolist()
                x[col] = x[col].apply(lambda element: replace_cat_woe(element, group_list, woe))

        if self.num_table_.keys():
            for col in self.num_table_.keys():
                woe = self.num_table_[col]["WoE"].tolist()
                break_list = self.num_table_[col][col].tolist()
                x[col] = x[col].apply(lambda element: replace_num_woe(element, break_list, woe))

        return x[self.keep_columns + list(self.information_values_.keys())]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)


