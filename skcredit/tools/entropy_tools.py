# coding:utf-8

import pandas as pd

from scipy.stats import entropy as scipy_entropy


def entropy(*args):
    return scipy_entropy(pd.concat(args, axis=1).value_counts(normalize=True), base=2)


def mis(x, y):
    # mutual information / information gain
    # I(x;y)   = H(x)  +  H(y)   - H(x,y)

    return entropy(x.to_frame(x.name)) + \
           entropy(y.to_frame(y.name)) - \
           entropy(x.to_frame(x.name), y.to_frame(y.name))


def cmi(x, y, z):
    # conditional mutual information
    # I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)

    return entropy(x.to_frame(x.name), z.to_frame(z.name)) + \
           entropy(y.to_frame(y.name), z.to_frame(z.name)) - \
           entropy(x.to_frame(x.name), y.to_frame(y.name), z.to_frame(z.name)) - \
           entropy(z.to_frame(z.name))
