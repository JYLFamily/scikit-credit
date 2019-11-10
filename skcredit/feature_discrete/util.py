# coding:utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
plt.style.use("ggplot")


def save_table(discrete, path):
    """
    :param discrete:
    :param path:
    :return:

    >>> save_table(discrete, path)
    """
    table = dict()
    table.update(discrete.num_table_)
    table.update(discrete.cat_table_)

    with pd.ExcelWriter(os.path.join(path, "table.xlsx")) as writer:
        for feature, table in table.items():
            table.to_excel(writer, sheet_name=feature[-30:], index=False)


def plot_importance(discretize):
    """
    :param discretize:
    :return:

    >>> plot_importance(discretize)
    >>> plt.show()
    """

    table = pd.DataFrame({
        "feature": list(discretize.information_values_.keys()),
        "information value": list(discretize.information_values_.values())
    })
    fig, ax = plt.subplots()
    ax = table.plot(
        x="feature",
        y="information value",
        kind="bar",
        ax=ax
    )
    ax.hlines(y=0.02, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles="dashed")
    ax.set_title(label="information value")

    return ax
