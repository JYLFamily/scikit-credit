# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from skcredit.tools import cat_bucket_to_string
from skcredit.feature_bucketer.BaseSplitND import BaseSplitND
np.random.seed(7)
pd.set_option("max_rows"   , None)
pd.set_option("max_columns", None)
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitCatND(BaseSplitND):
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

        from skcredit.feature_bucketer import WoEEncoder
        self.woe_encoder  = WoEEncoder(columns=self.columns, target=self.target)
        self.woe_encoder.fit(x, y)
        x = self.woe_encoder.transform(x)

        self._fit(  x,  y)

        return self

    def transform(self, x):
        x = self.woe_encoder.transform(x)

        return self._transform(x)

    def build_table(self ):
        table = self._table.copy(deep=True)

        table["Bucket"] = table["Bucket"].apply(
            lambda buckets: ', '.join([cat_bucket_to_string(bucket, self.woe_encoder.lookup[column])
                 for column, bucket in zip(self.columns, buckets)]))

        return table

    def build_image(self ):
        pass



