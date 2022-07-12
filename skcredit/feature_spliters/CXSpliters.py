# coding: utf-8

import warnings
import numpy  as np
import pandas as pd
from sklearn.utils import Bunch
from joblib import Parallel,  delayed
from tempfile      import TemporaryDirectory
from sklearn.pipeline import _fit_one,    _transform_one
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class CXSpliters(BaseEstimator,  TransformerMixin):
    def __init__(self, keep_columns, date_columns, transformers, iv_threshold, nthread, verbose):
        self.keep_columns = keep_columns
        self.date_columns = date_columns
        self.transformers = transformers
        self.iv_threshold = iv_threshold

        self.nthread = nthread
        self.verbose = verbose

    def fit(self, x,  y):
        with  TemporaryDirectory() as dirname:
            self.transformers = list(zip(
                Parallel(n_jobs=self.nthread, verbose=self.verbose, temp_folder=dirname )([
                    delayed(_fit_one)(
                        transformer,
                        x[columns ],
                        y,
                        weight=None
                    )    for transformer, columns in self.transformers
                ]),
                [columns for transformer, columns in self.transformers]
            ))

        return self

    def transform(self, x):
        z = pd.DataFrame(index=x.index,
            columns=self.keep_columns + self.date_columns + [" @ ".join(columns) for transformer, columns in
                    self.transformers if transformer.build_table()["IvS"].tolist()[-1] > self.iv_threshold])

        z[self.keep_columns] = x[self.keep_columns]
        z[self.date_columns] = x[self.date_columns]

        with  TemporaryDirectory() as dirname:
            z[[" @ ".join(columns) for transformer, columns in self.transformers if
                transformer.build_table()["IvS"].tolist()[-1] > self.iv_threshold]] = \
                pd.concat(
                    Parallel(n_jobs=self.nthread, verbose=self.verbose, temp_folder=dirname)([
                        delayed(_transform_one)(
                            transformer,
                            x[columns ],
                            None,
                            weight=None
                        )    for transformer, columns in self.transformers if
                        transformer.build_table()["IvS"].tolist()[-1] > self.iv_threshold
                    ]),
                    axis="columns"
            )

        return z

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)

        return self.transform(x)

    @property
    def named_transformers_table(self):

        return Bunch(**{" @ ".join(name): transformer.build_table() for transformer, name in self.transformers})

    @property
    def named_transformers_image(self):

        return Bunch(**{" @ ".join(name): transformer.build_image() for transformer, name in self.transformers})
