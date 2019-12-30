# coding:utf-8

import os
import numpy as np
import pandas as pd
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


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



