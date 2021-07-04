# coding:utf-8

import logging
import numpy  as np
import pandas as pd
from skcredit.feature_discretization.DiscreteTable import calc_cat_table, calc_num_table
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def merge_cat_table(x, col):
    # cat
    x_non = x.loc[x[col] != "missing"].copy(deep=True)
    x_mis = x.loc[x[col] == "missing"].copy(deep=True)
    x_non = x_non.reset_index(drop=True)
    x_mis = x_mis.reset_index(drop=True)

    # calc cat table
    table = calc_cat_table(x_non, x_mis, col)

    logging.info("{:<10} split complete !".format(col))

    return table


def merge_num_table(x, col):
    # num
    x_non = x.loc[x[col] != -999999.0].copy(deep=True)
    x_mis = x.loc[x[col] == -999999.0].copy(deep=True)
    x_non = x_non.reset_index(drop=True)
    x_mis = x_mis.reset_index(drop=True)

    # calc num table
    table = calc_num_table(x_non, x_mis, col)

    logging.info("{:<10} split complete !".format(col))

    return table


def force_cat_table(x, col, group_list):
    table = calc_cat_table(x, col, group_list)

    logging.info("{:<10} complete !".format(col))

    return table


def force_num_table(x, col, break_list):
    table = calc_num_table(x, col, break_list)

    logging.info("{:<10} complete !".format(col))

    return table


def replace_cat_woe(x, col, group_list, woe):
    non_mask = x != "missing"

    x_all_transform = np.zeros_like(x,           dtype=np.float64)
    x_non_transform = np.zeros_like(x[non_mask], dtype=np.float64)

    for i, section in enumerate(group_list[:-1]):
        mask = np.isin(x[non_mask], section)
        x_non_transform[mask] = woe[i]

    x_all_transform[ non_mask] = x_non_transform
    x_all_transform[~non_mask] = woe[-1]

    logging.info("{:<10} transform complete !".format(col))

    return x_all_transform


def replace_num_woe(x, col, break_list, woe):
    non_mask = x != -999999.0

    x_all_transform = np.zeros_like(x,           dtype=np.float64)
    x_non_transform = np.zeros_like(x[non_mask], dtype=np.float64)

    bins = [np.float_(section.end) for section in break_list[:-1]]
    idxs =  np.digitize(x=x[non_mask], bins=bins, right=True)

    for i in range(0, len(bins)):
        mask = idxs == i
        x_non_transform[mask] = woe[i]

    x_all_transform[ non_mask] = x_non_transform
    x_all_transform[~non_mask] = woe[-1]

    logging.info("{:<10} transform complete !".format(col))

    return x_all_transform


# def merge_cat_table_cross(X, col_1, col_2):
#     """
#     :param X:
#     :param col_1:
#     :param col_2:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     # cat to num
#     x_non = x.loc[(x[col_1] != "missing") & (x[col_2] != "missing")].copy(deep=True)
#     x_non = x_non.reset_index(drop=True)
#
#     weights_1 = (1 / (1 + np.exp(-(x_non.groupby(col_1).size() - 1))))
#     mapping_1 = (1 - weights_1) * x_non["target"].mean() + weights_1 * x_non.groupby(col_1)["target"].mean()
#     mapping_1 = mapping_1.to_dict()
#
#     weights_2 = (1 / (1 + np.exp(-(x_non.groupby(col_2).size() - 1))))
#     mapping_2 = (1 - weights_2) * x_non["target"].mean() + weights_2 * x_non.groupby(col_2)["target"].mean()
#     mapping_2 = mapping_2.to_dict()
#
#     x_non[col_1] = x_non[col_1].replace(mapping_1)
#     x_non[col_2] = x_non[col_2].replace(mapping_2)
#
#     # break list to group list
#     break_list_1, break_list_2, tree = dtree_split(x_non, col_1, col_2)
#     group_list_1, group_list_2 = [[] for _ in break_list_1], [[] for _ in break_list_2]
#
#     for k, v in mapping_1.items():
#         for idx, brk in break_list_1.items():
#             if v in brk:
#                 group_list_1[idx].append(k)
#
#     for k, v in mapping_2.items():
#         for idx, brk in break_list_2.items():
#             if v in brk:
#                 group_list_2[idx].append(k)
#
#     group_list_1 = pd.Series([", ".join(l) for l in group_list_1])
#     group_list_2 = pd.Series([", ".join(l) for l in group_list_2])
#
#     # calc cat table
#     table = calc_cat_table_cross(x, col_1, col_2, group_list_1, group_list_2)
#
#     logging.info("{:<10} @ {:<10} split complete !".format(col_1, col_2))
#
#     return table
#
#
# def merge_num_table_cross(X, col_1, col_2):
#     """
#     :param X:
#     :param col_1:
#     :param col_2:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     # num
#     x_non = x.loc[(x[col_1] != -9999) & (x[col_2] != -9999)].copy(deep=True)
#     x_non = x_non.reset_index(drop=True)
#
#     # break list
#     break_list_1, break_list_2, tree = dtree_split(x_non, col_1, col_2)
#
#     # calc num table
#     table = calc_num_table_cross(x, col_1, col_2, break_list_1, break_list_2)
#
#     logging.info("{:<10} @ {:<10} split complete !".format(col_1, col_2))
#
#     return table
#
#
# def force_cat_table_cross(X, col_1, col_2, group_list_1, group_list_2):
#     """
#     :param X:
#     :param col_1:
#     :param col_2:
#     :param group_list_1:
#     :param group_list_2:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     table = calc_cat_table_cross(x, col_1, col_2, group_list_1, group_list_2)
#
#     logging.info("{:<10} @ {:<10} complete !".format(col_1, col_2))
#
#     return table
#
#
# def force_num_table_cross(X, col_1, col_2, break_list_1, break_list_2):
#     """
#     :param X:
#     :param col_1:
#     :param col_2:
#     :param break_list_1:
#     :param break_list_2:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     table = calc_num_table_cross(x, col_1, col_2, break_list_1, break_list_2)
#
#     logging.info("{:<10} @ {:<10} complete !".format(col_1, col_2))
#
#     return table
#
#
# def replace_cat_woe_cross(X, col_1, col_2, group_list_1, group_list_2, woe):
#     """
#     :param X:
#     :param col_1
#     :param col_2
#     :param group_list_1:
#     :param group_list_2:
#     :param woe:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     def func(element):
#         if element[0] == "missing" or element[1] == "missing":
#             return woe[-1]
#         else:
#             for i, (l_1, l_2) in enumerate(zip(group_list_1, group_list_2)):
#                 if element[0] in l_1 and element[1] in l_2:
#                     return woe[i]
#
#     x = np.apply_along_axis(func, axis=1, arr=x.to_numpy())
#
#     logging.info("{:<10} @ {:<10} transform complete !".format(col_1, col_2))
#
#     return x
#
#
# def replace_num_woe_cross(X, col_1, col_2, break_list_1, break_list_2, woe):
#     """
#     :param X:
#     :param col_1
#     :param col_2
#     :param break_list_1:
#     :param break_list_2:
#     :param woe:
#     :return:
#     """
#     x = X.copy(deep=True)
#     del X
#     gc.collect()
#
#     def func(element):
#         if element[0] == -9999.0 or element[1] == -9999.0:
#             return woe[-1]
#         else:
#             for i, (l_1, l_2) in enumerate(zip(break_list_1, break_list_2)):
#                 if element[0] <= l_1.right and element[1] <= l_2.right:
#                     return woe[i]
#
#     x = np.apply_along_axis(func, axis=1, arr=x.to_numpy())
#
#     logging.info("{:<10} @ {:<10} transform complete !".format(col_1, col_2))
#
#     return x
