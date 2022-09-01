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
from tasks.train_tran import get_hparam_str

sns.set_context("paper", font_scale=1)

# %load_ext autoreload
# %autoreload 2

# data_args = argparse.Namespace(
#     dataset="mhp-1K-10", input_path="data/input", output_path="data/output_tran"
# )
# print(data_args)

# model_args = argparse.Namespace(
#     model="Tran", max_mean=100, n_bases=7, hidden_size=64, lr=0.001
# )
# print(model_args)

# for split_id in range(5):
#     plt.figure()
#     pred_A = np.loadtxt(
#         osp.join(
#             data_args.output_path,
#             data_args.dataset,
#             f"split_id={split_id}",
#             model_args.model,
#             get_hparam_str(model_args),
#             "scores_mat.txt",
#         )
#     )
#     pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

#     df = pd.DataFrame(pred_A)
#     sns.set()
#     ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

#     savefig(
#         ax.get_figure(),
#         osp.join(
#             data_args.output_path,
#             data_args.dataset,
#             f"{data_args.dataset}_{split_id}-{model_args.model}-score_mat.pdf",
#         ),
#     )


data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head1"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=1, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head1"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=2, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head1"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=3, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=4, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )



data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head5"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=1, tran_head=5
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head1"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=2, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

data_args = argparse.Namespace(
    dataset="pgem-1K-15", input_path="data/input", output_path="data/output_tran_head1"
)
print(data_args)

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=3, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )

model_args = argparse.Namespace(
    model="Tran", max_mean=20.0, n_bases=7, hidden_size=64, lr=0.0008, tran_layer=4, tran_head=1
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )
    pred_A = (pred_A - np.min(pred_A))/(np.max(pred_A) - np.min(pred_A))

    df = pd.DataFrame(pred_A)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-{model_args.tran_layer}-score_mat.pdf",
        ),
    )