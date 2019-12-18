# coding: utf-8

import gc
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.DiscreteImple import replace_cat_woe, replace_num_woe


class BaseDiscrete(BaseEstimator, TransformerMixin):
    def __init__(self,
            keep_columns, cat_columns, num_columns):
        self.keep_columns = keep_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns

        self.cat_table_ = dict()
        self.num_table_ = dict()

        self.cat_columns_ = None
        self.num_columns_ = None

        self.information_values_ = OrderedDict()

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.num_columns_ = list(self.num_table_.keys())
        self.cat_columns_ = list(self.cat_table_.keys())

        if len(self.cat_columns_) != 0:
            x = x.drop(list(set(self.cat_columns).difference(self.cat_columns_)), axis=1)

            for col in self.cat_table_.keys():
                woe = self.cat_table_[col]["WoE"].tolist()
                categories = self.cat_table_[col][col].tolist()
                x[col] = x[col].apply(lambda element: replace_cat_woe(element, categories, woe))

        if len(self.num_columns_) != 0:
            x = x.drop(list(set(self.num_columns).difference(self.num_columns_)), axis=1)

            for col in self.num_table_.keys():
                woe = self.num_table_[col]["WoE"].tolist()
                upper = self.num_table_[col]["Upper"].tolist()
                x[col] = x[col].apply(lambda element: replace_num_woe(element, upper, woe))

        return x[self.keep_columns + list(self.information_values_.keys())]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)


