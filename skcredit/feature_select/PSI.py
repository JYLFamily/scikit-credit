# encoding:utf-8

import gc
import numpy as np
import pandas as pd
from operator import *
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class PSI(BaseEstimator, TransformerMixin):
    def __init__(self, *, keep_columns):
        self.__keep_columns = keep_columns  # ["backdate"]
        self.__population_stability_index_x = None
        self.__population_stability_index_y = None

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        x["backdate"] = pd.to_datetime(x["backdate"])
        x["backdate"] = x["backdate"].apply(lambda element: element.strftime("%Y%m"))

        cols = x.columns.difference(pd.Index(self.__keep_columns))
        tdxs = np.sort(x["backdate"].unique().tolist())

        self.__population_stability_index_x = [[] for _ in cols]
        self.__population_stability_index_y = [[] for _ in cols]

        for idx, col in enumerate(cols):
            for tdx in range(sub(len(tdxs), 1)):
                # x psi
                dev = x.loc[x["backdate"] == tdxs[tdx], col].value_counts(normalize=True)
                val = x.loc[x["backdate"] == tdxs[add(tdx, 1)], col].value_counts(normalize=True)
                dev = dev[val.index]
                self.__population_stability_index_x[idx].append(np.sum((val - dev) * np.log(val / dev)))

                # x -> y psi
                dev = pd.concat([x[[col, "backdate"]], y.to_frame("target")], axis=1).loc[
                    x["backdate"] == tdxs[tdx], [col, "target"]].groupby(col)["target"].mean()
                val = pd.concat([x[[col, "backdate"]], y.to_frame("target")], axis=1).loc[
                    x["backdate"] == tdxs[add(tdx, 1)], [col, "target"]].groupby(col)["target"].mean()
                dev = dev[val.index]
                self.__population_stability_index_y[idx].append(np.sum((val - dev) * np.log(val / dev)))

        self.__population_stability_index_x = pd.DataFrame(self.__population_stability_index_x, index=cols)
        self.__population_stability_index_y = pd.DataFrame(self.__population_stability_index_y, index=cols)

        return self

    def transform(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        psi_x = self.__population_stability_index_x.apply(lambda element: np.max(element), axis=1)
        psi_y = self.__population_stability_index_y.apply(lambda element: np.max(element), axis=1)

        return x[
            list(set(psi_x[psi_x <= 0.1].index.tolist() + psi_y[psi_y <= 0.1].index.tolist())) + self.__keep_columns]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        return self.transform(X)