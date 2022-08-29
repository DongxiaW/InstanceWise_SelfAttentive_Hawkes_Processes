# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.utils.plotting import savefig
from tasks.train import get_hparam_str

sns.set_context("paper", font_scale=1)

# %load_ext autoreload
# %autoreload 2

# data_args = argparse.Namespace(
#     dataset="mhp-1K-10", input_path="data/input", output_path="data/output"
# )
# print(data_args)


# plt.figure()
# pred_A = np.loadtxt(
#     osp.join(
#         data_args.input_path,
#         data_args.dataset,
#         "infectivity.txt",
#     )
# )
# pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

# df = pd.DataFrame(pred_A)
# sns.set()
# ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

# savefig(
#     ax.get_figure(),
#     osp.join(
#         data_args.input_path,
#         data_args.dataset,
#         "infectivity.pdf",
#     ),
# )


data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_newexp"
)
print(data_args)


plt.figure()
pred_A = np.loadtxt(
    osp.join(
        data_args.input_path,
        data_args.dataset,
        "infectivity.txt",
    )
)
pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

df = pd.DataFrame(pred_A)
sns.set()
ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

savefig(
    ax.get_figure(),
    osp.join(
        data_args.input_path,
        data_args.dataset,
        "infectivity.pdf",
    ),
)