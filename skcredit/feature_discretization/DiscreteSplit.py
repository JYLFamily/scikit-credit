# coding:utf-8

import gc
import numpy as np
import pandas as pd
from collections import namedtuple
from sympy import Interval, Intersection
from sklearn.tree import DecisionTreeClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


def dleaf_rules(tree, feature_list):
    rule_tple = namedtuple("rule", ["feature", "interval"])
    rule_dict = dict()

    feature, threshold = tree.feature, tree.threshold
    children_l, children_r = tree.children_left, tree.children_right

    def recursions(node, rull):
        if children_l[node] == children_r[node]:
            rule_dict[node] = dict()

            for rule in rull:
                if rule.feature not in rule_dict[node].keys():
                    rule_dict[node][rule.feature] = rule.interval
                else:
                    rule_dict[node][rule.feature] = Intersection(rule_dict[node][rule.feature], rule.interval)
        else:
            rule_l = rule_tple(feature=feature_list[feature[node]], interval=Interval.Lopen(-999999, threshold[node]))
            rule_r = rule_tple(feature=feature_list[feature[node]], interval=Interval.open(threshold[node],  +999999))

            recursions(children_l[node], rull + [rule_l])
            recursions(children_r[node], rull + [rule_r])

    if children_l[0] == children_r[0]:
        rule_dict[0] = dict()

        for feature in feature_list:
            rule_dict[0][feature] = Interval.open(-999999, +999999)
    else:
        recursions(node=0, rull=[])

    rule_dict = pd.DataFrame.from_dict(rule_dict,   orient="index")

    for feature in feature_list:
        if feature not in rule_dict.columns:
            rule_dict[feature] = Interval.open(-999999,  +999999)
    rule_dict = rule_dict.fillna(Interval.open(-999999, +999999))
    rule_dict = rule_dict.reindex(columns=feature_list).reset_index(drop=True)

    return rule_dict


def dtree_split(X, col):
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = None

    # for min_impurity_decrease in np.arange(5e-4, 5e-2, 5e-3):
    #     clf = DecisionTreeClassifier(
    #         criterion="entropy", min_impurity_decrease=min_impurity_decrease, min_samples_leaf=0.05, random_state=7)
    #     clf.fit(x[[col]], x["target"])
    #     if x.groupby(clf.apply(x[[col]]))["target"].mean().is_monotonic:
    #         break
    clf = DecisionTreeClassifier(
        criterion="entropy", min_impurity_decrease=0., min_samples_leaf=0.05, random_state=7)
    clf.fit(x[[col]], x["target"])

    return dleaf_rules(clf.tree_, [col])[col]


def dtree_split_cross(X, col_1, col_2):
    x = X.copy(deep=True)
    del X
    gc.collect()

    clf = None

    for min_impurity_decrease in np.arange(5e-4, 5e-2, 5e-3):
        clf = DecisionTreeClassifier(
            criterion="entropy", min_impurity_decrease=min_impurity_decrease, min_samples_leaf=0.05, random_state=7)
        clf.fit(x[[col_1, col_2]], x["target"])
        if x.groupby(clf.apply(x[[col_1, col_2]]))["target"].mean().is_monotonic:
            break

    return dleaf_rules(clf.tree_, [col_1, col_2])[col_1], dleaf_rules(clf.tree_, [col_1, col_2])[col_2]
