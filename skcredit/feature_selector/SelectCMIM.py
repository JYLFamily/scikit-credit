# coding:utf-8

import warnings
import tempfile
import numpy  as  np
import pandas as  pd
import altair as alt
from skcredit.tools import mis
from skcredit.tools import cmi
from joblib import Parallel, delayed
from itertools   import combinations
from skcredit.feature_selector.BaseSelect import BaseSelect
np.random.seed(7)
pd.options.display.max_rows    = 999
pd.options.display.max_columns = 999
pd.set_option("display.unicode.east_asian_width" , True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
warnings.simplefilter(action="ignore", category=FutureWarning)


class SelectCMIM(BaseSelect):
    def __init__(self, keep_columns, date_columns, nums_columns, nthread, verbose):
        super(SelectCMIM, self).__init__(
                       keep_columns, date_columns, nums_columns, nthread, verbose)

        self.f_t_mi = None
        self.f_f_mi = None
        self.f_f_ci = None

    def fit(self, x, y=None):
        super(SelectCMIM, self).fit(x, y)

        self.f_t_mi = pd.Series(
                        index  =self.feature_columns, dtype=np.float64)
        self.f_f_mi = pd.DataFrame(np.zeros((self.feature_columns.shape[0], self.feature_columns.shape[0])),
                        columns=self.feature_columns, index=self.feature_columns, dtype=np.float64)
        self.f_f_ci = pd.DataFrame(np.zeros((self.feature_columns.shape[0], self.feature_columns.shape[0])),
                        columns=self.feature_columns, index=self.feature_columns, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmpdirname:
            f_t_mi_temp = Parallel(n_jobs=self.nums_columns, verbose=self.verbose, temp_folder=tmpdirname)(
                [delayed(mis)(x[col],              y) for col          in              self.feature_columns    ])

        with tempfile.TemporaryDirectory() as tmpdirname:
            f_f_mi_temp = Parallel(n_jobs=self.nums_columns, verbose=self.verbose, temp_folder=tmpdirname)(
                [delayed(mis)(x[col_i],  x[col_j]   ) for col_i, col_j in combinations(self.feature_columns, 2)])

        with tempfile.TemporaryDirectory() as tmpdirname:
            f_f_ci_temp = Parallel(n_jobs=self.nums_columns, verbose=self.verbose, temp_folder=tmpdirname)(
                [delayed(cmi)(x[col_i],  x[col_j], y) for col_i, col_j in combinations(self.feature_columns, 2)])

        for idx,  col           in enumerate(             self.feature_columns    ):
            self.f_t_mi.loc[col]          = f_t_mi_temp[idx]

        for idx, (col_i, col_j) in enumerate(combinations(self.feature_columns, 2)):
            self.f_f_mi.loc[col_i, col_j] = f_f_mi_temp[idx]
            self.f_f_mi.loc[col_j, col_i] = f_f_mi_temp[idx]

            self.f_f_ci.loc[col_i, col_j] = f_f_ci_temp[idx]
            self.f_f_ci.loc[col_j, col_i] = f_f_ci_temp[idx]

        self.feature_support[self.f_t_mi.argmax()] = True

        for _ in range(self.nums_columns - 1):
            f_t_mi_deselect = self.f_t_mi.loc[self.feature_columns[~self.feature_support]]
            f_f_mi_deselect = self.f_f_mi.loc[self.feature_columns[~self.feature_support],
                                              self.feature_columns[ self.feature_support]]
            f_f_ci_deselect = self.f_f_ci.loc[self.feature_columns[~self.feature_support],
                                              self.feature_columns[ self.feature_support]]

            scores = f_t_mi_deselect - np.maximum(0, f_f_mi_deselect.subtract(f_f_ci_deselect)).sum(axis = 1)

            self.feature_support[np.where(self.feature_columns == scores.idxmax())[0]]  =  True

        return self

    def plot_f_f_mi(self, selected=True):
        z    = (
            self.f_f_mi.loc[self.feature_columns[self.feature_support] if selected else self.feature_columns,
                            self.feature_columns[self.feature_support] if selected else self.feature_columns]
        ).to_numpy()

        x, y = np.meshgrid(
            self.feature_columns[self.feature_support] if selected else self.feature_columns,
            self.feature_columns[self.feature_support] if selected else self.feature_columns,
        )

        source = pd.DataFrame({
            "Feature-x": x.ravel(),
            "Feature-y": y.ravel(),
            "MIS":       z.ravel(),
        })

        figure = alt.Chart(
            source,
            title="Feature-Feature MIS"
        ).mark_rect().encode(
            x="Feature-x:N",
            y="Feature-y:N",
            color=alt.Color("MIS:Q", scale=alt.Scale(scheme="inferno")),
            tooltip=[
                alt.Tooltip("Feature-x:N",           title="Feature-x"),
                alt.Tooltip("Feature-y:N",           title="Feature-y")
            ]
        )

        return figure.show()

    def plot_f_f_ci(self, selected=True):
        z = (
            self.f_f_ci.loc[self.feature_columns[self.feature_support] if selected else self.feature_columns,
                            self.feature_columns[self.feature_support] if selected else self.feature_columns]
        ).to_numpy()

        x, y = np.meshgrid(
            self.feature_columns[self.feature_support] if selected else self.feature_columns,
            self.feature_columns[self.feature_support] if selected else self.feature_columns,
        )

        source = pd.DataFrame({
            "Feature-x": x.ravel(),
            "Feature-y": y.ravel(),
            "CIS":       z.ravel(),
        })

        figure = alt.Chart(
            source,
            title="Feature-Feature CIS"
        ).mark_rect().encode(
            x="Feature-x:N",
            y="Feature-y:N",
            color=alt.Color("CIS:Q", scale=alt.Scale(scheme="inferno")),
            tooltip=[
                alt.Tooltip("Feature-x:N", title="Feature-x"),
                alt.Tooltip("Feature-y:N", title="Feature-y")
            ]
        )

        return figure.show()
