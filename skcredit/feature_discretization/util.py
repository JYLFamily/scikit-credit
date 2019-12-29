# coding:utf-8

import os
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def cat_to_num(x, col):
    weights = (1 / (1 + np.exp(-(x.groupby(col).size() - 1))))
    mapping = (1 - weights) * x["target"].mean() + weights * x.groupby(col)["target"].mean()
    mapping = mapping.to_dict()

    x[col] = x[col].replace(mapping)

    return x, mapping


def save_table(discrete, path):
    """
    :param discrete:
    :param path:
    :return:

    >>> save_table(discrete, path)
    """
    table = dict()
    table.update(discrete.num_table_)
    table.update(discrete.cat_table_)

    with pd.ExcelWriter(os.path.join(path, "table.xlsx")) as writer:
        for feature, table in table.items():
            table.to_excel(writer, sheet_name=feature[-30:], index=False)



