# coding: utf-8

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class Tabular(BaseEstimator, TransformerMixin):
    def __init__(self, tabular):
        self.tabular = tabular

    @property
    def input(self):
        return self.tabular.drop(["target"], axis=1).copy(deep=True)

    @property
    def label(self):
        return self.tabular["target"].copy(deep=True)

    @property
    def trn_val_input(self):
        trn_input, val_input = train_test_split(self.input, train_size=0.75, random_state=7, shuffle=True)

        return trn_input.reset_index(drop=True), val_input.reset_index(drop=True)

    @property
    def trn_val_label(self):
        trn_label, val_label = train_test_split(self.label, train_size=0.75, random_state=7, shuffle=True)

        return trn_label.reset_index(drop=True), val_label.reset_index(drop=True)


class CTabular(BaseEstimator, TransformerMixin):
    def __init__(self, rep_columns):
        self.rep_columns = rep_columns

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        for k, v in self.rep_columns.items():
            for i in v:
                x[k] = x[k].replace({i: np.nan})

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)


class FTabular(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns, cat_columns, num_columns):
        self.tim_columns = tim_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns

        self.cat_value_ = dict()
        self.num_value_ = dict()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if self.cat_columns:
            x[self.cat_columns] = x[self.cat_columns].fillna("missing").astype(str)

            for col in self.cat_columns:
                stats = x[col].value_counts(normalize=True)
                if 1 - stats.max() >= 0.05:
                    self.cat_value_[col] = stats.to_dict()

        if self.num_columns:
            x[self.num_columns] = x[self.num_columns].fillna(- 9999.00)

            for col in self.num_columns:
                stats = x[col].value_counts(normalize=True)
                if 1 - stats.max() >= 0.05:
                    self.num_value_[col] = -9999.

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        x = x[self.tim_columns + list(self.cat_value_.keys()) + list(self.num_value_.keys())]

        if self.tim_columns:
            x[self.tim_columns[0]] = pd.to_datetime(x[self.tim_columns[0]])

        if self.cat_value_.keys():
            x[list(self.cat_value_.keys())] = x[list(self.cat_value_.keys())].fillna("missing").astype(str)

            for col in self.cat_value_.keys():
                to_rep = ("missing" if "missing" in self.cat_value_[col].keys()
                          else list(self.cat_value_[col].keys())[0])
                x[col] = x[col].apply(
                    lambda element: element if element in self.cat_value_[col].keys() else to_rep)

        if self.num_value_.keys():
            for col in self.num_value_.keys():
                x[col] = x[col].fillna(self.num_value_[col])

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)


