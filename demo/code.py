# encoding: utf-8

import os
import gc
import yaml
import numpy as np
import pandas as pd
from skcredit.models import cv, LRClassifier
from skcredit.feature_selection import SelectBin
from skcredit.feature_selection import SelectVif
from skcredit.feature_discretization import Discrete
from skcredit.feature_preprocessing import PreProcess
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    tra = pd.read_csv(os.path.join(config["path"], config["dataset"]["tra"]))
    tes = pd.read_csv(os.path.join(config["path"], config["dataset"]["tes"]))

    tra = tra.drop(columns=config["columns"]["drop"])
    tes = tes.drop(columns=config["columns"]["drop"]).tail(2000)

    tra_feature, tra_label = (tra.drop(columns=config["columns"]["target"]).copy(deep=True),
                              tra[config["columns"]["target"]].copy(deep=True))
    tes_feature, tes_label = (tes.drop(columns=config["columns"]["target"]).copy(deep=True),
                              tes[config["columns"]["target"]].copy(deep=True))
    del tra, tes
    gc.collect()

    from pprint import pprint
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("preprocess", PreProcess(
                        keep_columns=[],
                        cat_columns=[],
                        num_columns=tra_feature.columns.tolist())),
        ("discrete", Discrete(
                        keep_columns=[],
                        cat_columns=[],
                        num_columns=tra_feature.columns.tolist(),
                        merge_bin=0.05,
                        information_value_threshold=0.002)),
        ("sbin", SelectBin(keep_columns=[])),
        ("svif", SelectVif(keep_columns=[], rs_threshold=0.8)),
        ("model", LRClassifier(keep_columns=[], c=1, random_state=7)),
    ])

    result = cv([0.25, 0.5, 1.], pipeline, tra_feature, tra_label, 7)
    pprint(result)
