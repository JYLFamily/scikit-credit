# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.feature_bucketer.SplitND import SplitND
from skcredit.tools import cat_bucket_to_string, CatEncoder
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitCatND(SplitND):
    def __init__(self,
                 column,
                 target,
                 min_bin_cnt_negative=75,
                 min_bin_cnt_positive=75,
                 min_information_value_split_gain=0.015):
        super().__init__(
            column,
            target,
            min_bin_cnt_negative,
            min_bin_cnt_positive,
            min_information_value_split_gain)

        self.cat_encoder = None

    def fit(self,   x,  y):
        super().fit(x,  y)

        self.cat_encoder =  CatEncoder(column=self.column, target=self.target)
        self.cat_encoder.fit(x, y)
        x = self.cat_encoder.transform(x)

        self._fit(  x,  y)

        return self

    def transform(self, x):
        x = self.cat_encoder.transform(x)

        return self._transform(x)

    def build_table(self ):
        table = self._table.copy(deep=True)

        table["Column"] = table["Column"].apply(lambda  columns:   f"FEATURE({', '.join(columns)})")
        table["Bucket"] = table["Bucket"].apply(
            lambda buckets: ', '.join([cat_bucket_to_string(bucket, self.cat_encoder.lookup[column])
                 for column, bucket in zip(self.column, buckets)]))

        return table


def binning_cat(x, y, column, target):
    scnd = SplitCatND(column, target)
    scnd.fit(x, y)

    return scnd


def replace_cat(x, scnd):

    return scnd.transform(x)



