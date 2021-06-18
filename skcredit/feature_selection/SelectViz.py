# coding: utf-8

import gc
import logging
import numpy  as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import filterfalse
from heapq import heappop, heappush
from factor_analyzer import Rotator
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
logging.basicConfig(format="[%(asctime)s]-[%(filename)s]-[%(levelname)s]-[%(message)s]", level=logging.INFO)


def get_bunch(X):
    # pca
    pca = PCA(n_components=2, random_state=7)
    raw_principal_component = pca.fit_transform(X)
    # rot
    rot = Rotator(method="quartimax")
    rot_principal_component = np.dot(X, rot.fit_transform(pca.components_.T))

    # bunch
    bunch = Bunch(
        split_feature=X.columns.tolist(),
        eigen_value_1=pca.explained_variance_[0],
        eigen_value_2=pca.explained_variance_[1],
        explained_variance_ratio=sum(pca.explained_variance_ratio_[:1]),
        raw_principal_component_1=raw_principal_component[:, 0],
        raw_principal_component_2=raw_principal_component[:, 1],
        rot_principal_component_1=rot_principal_component[:, 0],
        rot_principal_component_2=rot_principal_component[:, 1]
    )

    return bunch


class Bunch(object):
    def __init__(self,
                 split_feature,
                 eigen_value_1,
                 eigen_value_2,
                 explained_variance_ratio,
                 raw_principal_component_1,  # raw principal component
                 raw_principal_component_2,
                 rot_principal_component_1,  # rot principal component
                 rot_principal_component_2):
        self.split_feature = split_feature
        self.eigen_value_1 = eigen_value_1
        self.eigen_value_2 = eigen_value_2
        self.explained_variance_ratio = explained_variance_ratio
        self.raw_principal_component_1 = raw_principal_component_1
        self.raw_principal_component_2 = raw_principal_component_2
        self.rot_principal_component_1 = rot_principal_component_1
        self.rot_principal_component_2 = rot_principal_component_2

    def __lt__(self, *args, **kwargs):
        if self.eigen_value_2 == args[0].eigen_value_2:
            return self.split_feature[0] < self.split_feature[0]
        return self.eigen_value_2 > args[0].eigen_value_2


class SelectViz(BaseEstimator, TransformerMixin):
    def __init__(self, tim_columns):
        self.tim_columns = tim_columns
        self.cluster_heapque_ = list()
        self.cluster_sorteds_ = list()

    def fit(self, X, y=None):
        x = X.copy(deep=True)
        del X
        gc.collect()

        self.cluster_heapque_ = list()
        heappush(
            self.cluster_heapque_,
            get_bunch(x[[col for col in x.columns if col not in self.tim_columns]])
        )

        while self.cluster_heapque_[0].eigen_value_2 >= 1:
            # assign feature to bunch
            split_feature_1 = list()
            split_feature_2 = list()
            for feature in self.cluster_heapque_[0].split_feature:
                corr_1 = abs(pearsonr(x[feature].to_numpy(), self.cluster_heapque_[0].rot_principal_component_1)[0])
                corr_2 = abs(pearsonr(x[feature].to_numpy(), self.cluster_heapque_[0].rot_principal_component_2)[0])

                if corr_1 > corr_2:
                    split_feature_1.append(feature)
                if corr_1 < corr_2:
                    split_feature_2.append(feature)

            # heappops
            heappop(self.cluster_heapque_)

            # heappush
            if len(split_feature_1) >= 2:
                heappush(
                    self.cluster_heapque_,
                    get_bunch(x[split_feature_1])
                )
            elif len(split_feature_1) == 1:
                heappush(
                    self.cluster_heapque_,
                    Bunch(
                        split_feature=split_feature_1,
                        eigen_value_1=1,
                        eigen_value_2=0,
                        explained_variance_ratio=1,
                        raw_principal_component_1=x[split_feature_1].to_numpy().reshape(-1,),
                        raw_principal_component_2=None,
                        rot_principal_component_1=None,
                        rot_principal_component_2=None,
                    ))
            elif len(split_feature_1) == 0:
                pass

            if len(split_feature_2) >= 2:
                heappush(
                    self.cluster_heapque_,
                    get_bunch(x[split_feature_2])
                )
            elif len(split_feature_2) == 1:
                heappush(
                    self.cluster_heapque_,
                    Bunch(
                        split_feature=split_feature_2,
                        eigen_value_1=1,
                        eigen_value_2=0,
                        explained_variance_ratio=1,
                        raw_principal_component_1=x[split_feature_2].to_numpy().reshape(-1,),
                        raw_principal_component_2=None,
                        rot_principal_component_1=None,
                        rot_principal_component_2=None,
                    )
                )
            elif len(split_feature_2) == 0:
                pass

        self.cluster_sorteds_ = []
        while len(self.cluster_heapque_) != 0:
            self.cluster_sorteds_.append(heappop(self.cluster_heapque_))

        return self

    def var_table(self):
        var_table = pd.DataFrame(columns=["Cluster", "N_Var", "E_V_1", "E_V_2", "Exp_Var_Prop"])

        for clu_idx, bunch_own in enumerate(self.cluster_sorteds_):
            var_table = var_table.append(
                pd.DataFrame([[clu_idx,
                               len(bunch_own.split_feature),
                               bunch_own.eigen_value_1,
                               bunch_own.eigen_value_2,
                               bunch_own.explained_variance_ratio]],
                             columns=var_table.columns),
                ignore_index=True
            )

        return var_table

    def rsq_table(self, X):
        x = X.copy(deep=True)
        del X
        gc.collect()

        rsq_table = pd.DataFrame(columns=["Cluster", "Feature", "RS_Own", "RS_Ncl", "RS_Ratio"])

        for clu_idx, bunch_own in enumerate(self.cluster_sorteds_):
            for feature in bunch_own.split_feature:
                rs_own =  pearsonr(x[feature].to_numpy(), bunch_own.raw_principal_component_1)[0] ** 2
                rs_lst = [pearsonr(x[feature].to_numpy(), bunch_oth.raw_principal_component_1)[0] ** 2
                          for bunch_oth in  self.cluster_sorteds_ if bunch_oth != bunch_own]
                rs_ncl = max(rs_lst) if len(rs_lst) != 0 else 0

                rsq_table = rsq_table.append(
                    pd.DataFrame([[clu_idx, feature, rs_own, rs_ncl, (1 - rs_own) / (1 - rs_ncl)]],
                                 columns=rsq_table.columns),
                    ignore_index=True
                )

        return rsq_table.sort_values(["Cluster", "RS_Ratio"], ascending=[True, True])

    def transform(self, X):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        pass
