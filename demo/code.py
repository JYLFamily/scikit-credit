# encoding: utf-8

import os
import gc
import yaml
import numpy as np
import pandas as pd
from skcredit.models import LRClassifier
from skcredit.feature_selection import SelectBin
from skcredit.feature_selection import SelectVif
from skcredit.feature_discretization import save_table, Discrete
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

    discrete = Discrete(
        keep_columns=[],
        cat_columns=[],
        num_columns=tra_feature.columns.tolist(),
        merge_bin=0.05,
        information_value_threshold=0.002
    )
    discrete.fit(tra_feature, tra_label)
    tra_feature = discrete.transform(tra_feature)
    tes_feature = discrete.transform(tes_feature)
    save_table(discrete, config["path"])

    sbin = SelectBin(keep_columns=[])
    sbin.fit(tra_feature, tra_label)
    tra_feature = sbin.transform(tra_feature)
    tes_feature = sbin.transform(tes_feature)

    svif = SelectVif(keep_columns=[], vif_threshold=2)
    svif.fit(tra_feature, tra_label)
    tra_feature = svif.transform(tra_feature)
    tes_feature = svif.transform(tes_feature)

    model = LRClassifier(keep_columns=[], c=1, random_state=7)
    model.fit(tra_feature, tra_label)

    print(model.score(tra_feature, tra_label))
    print(model.score(tes_feature, tes_label))
