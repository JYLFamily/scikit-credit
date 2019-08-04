# coding:utf-8

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from path import Path
from operator import itemgetter
from sklearn.base import BaseEstimator, TransformerMixin
from skcredit.feature_discretize.DiscretizeUtil import merge_num_table, merge_cat_table
from skcredit.feature_discretize.DiscretizeUtil import replace_num_woe, replace_cat_woe
np.random.seed(7)
plt.style.use("ggplot")


def save_table(binning, path):
    """
    :param binning:
    :param path:
    :return:

    >>> save_table(binning, path)
    """
    table = dict()
    table.update(binning.num_table_)
    table.update(binning.cat_table_)

    with pd.ExcelWriter(Path(path) / "table.xlsx") as writer:
        for feature, table in table.items():
            table.to_excel(writer, sheet_name=feature[-30:], index=False)


def plot_importance(binning):
    """
    :param binning:
    :return:

    >>> plot_importance(binning)
    >>> plt.show()
    """

    table = pd.DataFrame({
        "feature": list(binning.information_values_.keys()),
        "information value": list(binning.information_values_.values())
    })
    fig, ax = plt.subplots()
    ax = table.plot(
        x="feature",
        y="information value",
        kind="bar",
        ax=ax
    )
    ax.hlines(y=0.01, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles="dashed")
    ax.set_title(label="information value")

    return ax


class Discretize(BaseEstimator, TransformerMixin):
    def __init__(self, *, cat_columns, num_columns, threshold=0.01):
        self.__cat_columns = cat_columns
        self.__num_columns = num_columns
        self.__threshold = threshold

        self.cat_table_ = dict()
        self.num_table_ = dict()

        self.cat_columns_ = None
        self.num_columns_ = None
        
        self.information_values_ = dict()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        if self.__cat_columns is not None:
            x[self.__cat_columns] = x[self.__cat_columns].fillna("missing").astype(str)

            for col in self.__cat_columns:
                table = merge_cat_table(pd.concat([x[[col]], y.to_frame("target")], axis=1), col)

                if table["IV"].sum() > self.__threshold:
                    self.cat_table_[col] = table
                    self.information_values_[col] = self.cat_table_[col]["IV"].sum()
                else:
                    continue

        if self.__num_columns is not None:
            x[self.__num_columns] = x[self.__num_columns].fillna(-9999.0)

            for col in self.__num_columns:
                table = merge_num_table(pd.concat([x[[col]], y.to_frame("target")], axis=1), col)

                if table["IV"].sum() > self.__threshold:
                    self.num_table_[col] = table
                    self.information_values_[col] = self.num_table_[col]["IV"].sum()
                else:
                    continue

        self.information_values_ = dict(
            sorted(self.information_values_.items(), key=itemgetter(1), reverse=True))

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.num_columns_ = list(self.num_table_.keys())
        self.cat_columns_ = list(self.cat_table_.keys())

        if len(self.cat_columns_) != 0:
            x = x.drop(list(set(self.__cat_columns).difference(self.cat_columns_)), axis=1)
            x[self.cat_columns_] = x[self.cat_columns_].fillna("missing").astype(str)

            for col in self.cat_table_.keys():
                woe = self.cat_table_[col]["WoE"].tolist()
                categories = self.cat_table_[col][col].tolist()
                x[col] = x[col].apply(lambda element: replace_cat_woe(element, categories, woe))

        if len(self.num_columns_) != 0:
            x = x.drop(list(set(self.__num_columns).difference(self.num_columns_)), axis=1)
            x[self.num_columns_] = x[self.num_columns_].fillna(-9999.0)

            for col in self.num_table_.keys():
                woe = self.num_table_[col]["WoE"].tolist()
                upper = self.num_table_[col]["Upper"].tolist()
                x[col] = x[col].apply(lambda element: replace_num_woe(element, upper, woe))

        return x

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)