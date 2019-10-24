# encoding: utf-8

import os
import yaml
import numpy as np
import pandas as pd
from skcredit.features.Discretize import save_table, Discretize
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)

    train = pd.read_csv(os.path.join(config["path"], config["dataset"]["tra"]), na_values=[-2], encoding="GBK")
    train_feature, train_label = (
        train.drop(config["columns"]["target"], axis=1), train[config["columns"]["target"]].copy(deep=True))

    discretize = Discretize(
        cat_columns=config["feature dis"]["cat cols"],
        num_columns=config["feature dis"]["num cols"],
        keep_columns=config["columns"]["keep"],
        merge_threshold=0.2,
        min_samples_bins=0.05,
        threshold=0.02
    )
    discretize.fit(train_feature, train_label.squeeze())
    train_feature = discretize.transform(train_feature)
    save_table(discretize, config["path"])

    # import statsmodels.api as sm
    # logit_mod = sm.Logit(train_label, sm.add_constant(train_feature, prepend=False))
    # logit_res = logit_mod.fit()
    # print(logit_res.summary())
    # print(logit_res.pvalues)


