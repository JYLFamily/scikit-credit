# encoding: utf-8

import gc
import os
import yaml
import numpy as np
import pandas as pd
from skcredit.feature_discretize.Discretize import Discretize
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.BaseLoader)
    train = pd.read_csv(os.path.join(config["path"], config["dataset"]["tra"]), encoding="GBK")
    test = pd.read_csv(os.path.join(config["path"], config["dataset"]["tes"]), encoding="GBK")

    train_output = train[config["columns"]["target"]].copy(deep=True)
    test_output = test[config["columns"]["target"]].copy(deep=True)

    train_input = train.drop(config["columns"]["drop"], axis=1).copy(deep=True)
    test_input = test.drop(config["columns"]["drop"], axis=1).copy(deep=True)

    del train, test
    gc.collect()

    discretize = Discretize(
        cat_columns=config["feature dis"]["cat cols"],
        num_columns=config["feature dis"]["num cols"],
        keep_columns=config["columns"]["keep"])
    discretize.fit(train_input, train_output)
    train_input = discretize.transform(train_input)
    test_input = discretize.transform(test_input)
