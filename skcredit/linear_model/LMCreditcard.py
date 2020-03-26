# coding:utf-8

import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def calc_table(lmclassifier, tables, col):
    table = copy.deepcopy(tables[col][[col, "WoE"]])
    table["Coefficients"] = lmclassifier.coeff_[col]
    table["PartialScore"] = - lmclassifier.coeff_[col] * lmclassifier.b_ * table["WoE"]

    return table


class LMCreditcard(object):
    def __init__(self, discrete, lmclassifier):
        self.discrete = discrete
        self.lmclassifier = lmclassifier

    def __call__(self, *args, **kwargs):
        tables = dict()
        tables.update(self.discrete.cat_table_)
        tables.update(self.discrete.num_table_)

        with Pool(mp.cpu_count() - 2) as pool:
            result = dict(zip(self.lmclassifier.feature_subsets_, pool.starmap(
                calc_table,
                [(self.lmclassifier, tables, col) for col in self.lmclassifier.feature_subsets_])))

        return result
