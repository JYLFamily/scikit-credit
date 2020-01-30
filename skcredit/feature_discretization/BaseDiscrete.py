# coding: utf-8

import os
import gc
import pandas as pd
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretization.DiscreteImple import replace_cat_woe, replace_num_woe


class BaseDiscrete(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns
        self.cat_columns_ = None
        self.num_columns_ = None

        self.cat_table_ = dict()
        self.num_table_ = dict()
        self.cat_table_cross_ = dict()
        self.num_table_cross_ = dict()

        self.cat_value_ = OrderedDict()
        self.num_value_ = OrderedDict()
        self.cat_value_cross_ = OrderedDict()
        self.num_value_cross_ = OrderedDict()

        self.information_value_ = OrderedDict()
        self.information_value_cross_ = OrderedDict()

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        x = x[self.tim_columns + list(self.information_value_.keys())]

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

        return x[self.tim_columns + list(self.information_value_.keys())]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)

    def save_order(self, path):
        order = pd.DataFrame.from_dict(self.information_value_, orient="index", columns=["IV"])
        order = order.reset_index().rename(columns={"index": "feature"})

        with pd.ExcelWriter(os.path.join(path, "order.xlsx")) as writer:
            order.to_excel(writer, index=False)

    def save_table(self, path):
        table = dict()
        table.update(self.num_table_)
        table.update(self.cat_table_)

        with pd.ExcelWriter(os.path.join(path, "table.xlsx")) as writer:
            for feature, table in table.items():
                table.to_excel(writer, sheet_name=feature[-30:], index=False)

    def save_order_cross(self, path):
        order = pd.DataFrame.from_dict(self.information_value_cross_, orient="index", columns=["IV"])
        order = order.reset_index().rename(columns={"index": "feature"})

        with pd.ExcelWriter(os.path.join(path, "order_cross.xlsx")) as writer:
            order.to_excel(writer, index=False)

    def save_table_cross(self, path):
        table = dict()
        table.update(self.num_table_cross_)
        table.update(self.cat_table_cross_)

        with pd.ExcelWriter(os.path.join(path, "table_cross.xlsx")) as writer:
            for feature, table in table.items():
                table.to_excel(writer, sheet_name=feature[-30:], index=False)

