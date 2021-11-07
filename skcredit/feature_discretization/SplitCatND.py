# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from feature_engine.encoding import WoEEncoder
from skcredit.feature_discretization.SplitND import SplitND
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitCatND(SplitND):
    def __init__(self,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

        self.woe_encoder = None

    def fit(self,   x,  y):
        super().fit(x,  y)

        self.woe_encoder  =  WoEEncoder()
        self.woe_encoder.fit(x, y)
        x = self.woe_encoder.transform(x)

        self._fit(  x,  y)

        return self

    def transform(self, x):
        x = x.fillna("missing")
        x = self.woe_encoder.transform(x)

        return self._transform(x)


def binning_cat(x,    y):
    snnd = SplitCatND( )
    snnd.fit(x, y)
    return snnd


def replace_cat(x, scnd):
    return scnd.transform(x)


if __name__ == "__main__":
    application_train = pd.read_csv(
        "C:\\Users\\P1352\\Desktop\\application_train.csv",
        usecols=["NAME_CONTRACT_TYPE", "HOUSETYPE_MODE", "TARGET"])
    scnd = SplitCatND()
    scnd.fit(application_train[["NAME_CONTRACT_TYPE", "HOUSETYPE_MODE"]], application_train["TARGET"])
    print(scnd._table)

