# coding:utf-8

import numpy  as np
import pandas as pd
import lightgbm as lgb
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


INF = 999999


def dtree_leafs(t, *columns):
    from sympy import Interval, Intersection
    from collections import defaultdict, OrderedDict
    rule_dict = defaultdict(OrderedDict)

    t.set_index(t["node_index"],        inplace=True)
    t.drop(["node_index"], axis="columns", inplace=True)

    def recursions(node, path):
        if t["left_child"][node] == t["right_child"][node]:
            for col in columns:
                from functools import reduce
                rule_dict[col][int(node[-1])] = reduce(
                    Intersection, path[col],  Interval.Lopen(-INF, +INF))
            return

        path[t["split_feature"][node]].append(Interval.Lopen(-INF, t["threshold"][node]))
        recursions( t["left_child"][node], path)
        path[t["split_feature"][node]].pop()

        path[t["split_feature"][node]].append(Interval.Lopen(t["threshold"][node], +INF))
        recursions(t["right_child"][node], path)
        path[t["split_feature"][node]].pop()

    recursions(node="0-S0", path=defaultdict(list))

    rule_data = pd.DataFrame({col: pd.Series(rule_dict[col]) for col in columns})
    rule_data = rule_data.reindex(columns=columns)

    return rule_data


def dtree_split(x, *columns):
    from scipy.stats import pearsonr
    spliter = lgb.train(
        params={
            "seed": 7, "num_threads": 1,
            "verbosity": -1,
            "objective": "binary",   "num_leaves": 10,
            "min_data_in_leaf": int(x.shape[0] // 20),
            "monotone_constraints": [1 if pearsonr(x[col], x["target"])[0] > 0 else -1 for col in columns],
        },
        train_set=lgb.Dataset(x[list(columns)], label=x["target"]),
        num_boost_round=1,
    )

    return spliter, dtree_leafs(spliter.trees_to_dataframe(), *columns)

