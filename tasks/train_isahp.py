import argparse
import os
import os.path as osp
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.models.eventseqdataset import EventSeqDataset
from pkg.models.isahp import InstancewiseSelfAttentiveHawkesProcesses
from pkg.utils.argparser.training import add_subparser_arguments
from pkg.utils.evaluation import eval_fns
from pkg.utils.logging import get_logger, init_logging
from pkg.utils.misc import (
    Timer,
    compare_metric_value,
    export_csv,
    export_json,
    get_freer_gpu,
    makedirs,
    set_rand_seed,
)

from pkg.utils.torch import split_dataloader, convert_to_bucketed_dataloader


def get_parser():
    parser = argparse.ArgumentParser(description="Training different models. ")
    subparsers = parser.add_subparsers(
        description="Supported models", dest="model"
    )
    for model in ["ISAHP"]:
        add_subparser_arguments(model, subparsers)

    return parser


def get_model(args, n_types):
    if args.model == "ISAHP":
        model = InstancewiseSelfAttentiveHawkesProcesses(n_types=n_types, **vars(args))
    else:
        raise ValueError(f"Unsupported model={args.model}")

    return model


def get_device(cuda, dynamic=True):
    if torch.cuda.is_available() and args.cuda:
        if dynamic:
            device = torch.device("cuda: 0")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_hparam_str(args):
    if args.model == "ISAHP":
        hparams = ["batch_size", "hidden_size", "num_head", "l1_reg", "type_reg", "lr"]
    else:
        hparams = []

    return ",".join("{}={}".format(p, getattr(args, p)) for p in hparams)


def train_nn_models(model, event_seqs, args):

    train_dataloader = DataLoader(
        EventSeqDataset(event_seqs), **dataloader_args
    )

    train_dataloader, valid_dataloader = split_dataloader(
        train_dataloader, 8 / 9
    )
    if "bucket_seqs" in args and args.bucket_seqs:
        train_dataloader = convert_to_bucketed_dataloader(
            train_dataloader, key_fn=len
        )
    valid_dataloader = convert_to_bucketed_dataloader(
        valid_dataloader, key_fn=len, shuffle_same_key=False
    )

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.lr
    )

    model.train()
    best_metric = float("nan")

    for epoch in range(args.epochs):
        train_metrics, valid_metrics = model.train_epoch(
            train_dataloader,
            optimizer,
            valid_dataloader,
            device=device,
            **vars(args),
        )

        msg = f"[Training] Epoch={epoch}"
        for k, v in train_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        logger.info(msg)
        msg = f"[Validation] Epoch={epoch}"
        for k, v in valid_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        logger.info(msg)

        if compare_metric_value(
            valid_metrics[args.tune_metric].avg, best_metric, args.tune_metric
        ):
            if epoch > args.epochs // 2:
                logger.info(f"Found a better model at epoch {epoch}.")
            best_metric = valid_metrics[args.tune_metric].avg
            torch.save(model.state_dict(), osp.join(output_path, "model.pt"))

    model.load_state_dict(torch.load(osp.join(output_path, "model.pt")))

    return model


def eval_nll(model, event_seqs, args):
    if args.model in ["ISAHP"]:

        dataloader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **dataloader_args
        )

        metrics = model.evaluate(dataloader, device=device)
        logger.info(
            "[Test]"
            + ", ".join(f"{k}={v.avg:.4f}" for k, v in metrics.items())
        )
        nll = metrics["nll"].avg.item()

    else:
        nll = float("nan")
        print("not supported yet")

    return nll


def predict_next_event(model, event_seqs, args):
    if args.model == "ISAHP":
        dataloader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **dataloader_args
        )
        event_seqs_pred_type, event_seqs_truth_type = model.predict_next_event_type(dataloader, device=device)
        return event_seqs_pred_type, event_seqs_truth_type
    else:
        print(
            "Predicting next event is not supported for "
            f"model={args.model} yet."
        )
        event_seqs_pred = None

    return event_seqs_pred


def get_infectivity_matrix(model, event_seqs, args):

    if args.model in ["ISAHP"]:
        _dataloader_args = dataloader_args.copy()
        if "attr_batch_size" in args and args.attr_batch_size:
            _dataloader_args.update(batch_size=args.attr_batch_size)

        dataloader = DataLoader(
            EventSeqDataset(event_seqs), **_dataloader_args
        )
        dataloader = convert_to_bucketed_dataloader(dataloader, key_fn=len)
        infectivity = model.get_infectivity(dataloader, device, **vars(args))
    else:
        print(
            "Computing infectivity matrix is not supported for "
            f"model={args.model} yet."
        )
        infectivity = None

    return infectivity


if __name__ == "__main__":

    args = get_parser().parse_args()
    assert args.model is not None, "`model` needs to be specified."

    output_path = osp.join(
        args.output_dir,
        args.dataset,
        f"split_id={args.split_id}",
        args.model,
        get_hparam_str(args),
    )
    makedirs([output_path])

    # initialization
    set_rand_seed(args.rand_seed, args.cuda)
    init_logging(output_path)
    logger = get_logger(__file__)

    logger.info(args)
    export_json(vars(args), osp.join(output_path, "config.json"))

    # load data
    input_path = osp.join(args.input_dir, args.dataset)
    data = np.load(osp.join(input_path, "data.npz"), allow_pickle=True)
    n_types = int(data["n_types"])
    event_seqs = data["event_seqs"]
    train_event_seqs = event_seqs[data["train_test_splits"][args.split_id][0]]
    test_event_seqs = event_seqs[data["train_test_splits"][args.split_id][1]]
    # sorted test_event_seqs by their length
    test_event_seqs = sorted(test_event_seqs, key=lambda seq: len(seq))

    if osp.exists(osp.join(input_path, "infectivity.txt")):
        A_true = np.loadtxt(osp.join(input_path, "infectivity.txt"))
    else:
        A_true = None

    with Timer("Training model"):
        # define model
        model = get_model(args, n_types)

        if args.model in ["ISAHP"]:
            dataloader_args = {
                "batch_size": args.batch_size,
                "collate_fn": EventSeqDataset.collate_fn,
                "num_workers": args.num_workers,
            }
            device = get_device(args.cuda)
            print('device:',device, flush=True)

            model = model.to(device)
            model = train_nn_models(model, train_event_seqs, args)

    # evaluate nll
    results = {}
    with Timer("Evaluate negative log-likelihood"):
        results["nll"] = eval_nll(model, test_event_seqs, args)
        print("nll", results["nll"])

    # evaluate next event prediction
    if not args.skip_pred_next_event:
        with Timer("Predict the next event"):
            event_seqs_pred_type, event_seqs_truth_type = predict_next_event(model, test_event_seqs, args)

            event_seqs_pred_type1 = [seq[:] for seq in event_seqs_pred_type]
            event_seqs_truth_type1 = [seq[:] for seq in event_seqs_truth_type]
            
            event_seqs_pred = np.concatenate(event_seqs_pred_type1)
            event_seqs_truth = np.concatenate(event_seqs_truth_type1)

            if event_seqs_pred is not None:
                for metric_name in ["acc"]:
                    results[metric_name] = eval_fns[metric_name](
                        event_seqs_truth, event_seqs_pred
                    )
                    # print(metric_name, results[metric_name])

    # evaluate infectivity matrix
    if not args.skip_eval_infectivity:
        A_pred = get_infectivity_matrix(model, event_seqs, args)
        np.savetxt(osp.join(output_path, "scores_mat.txt"), A_pred)

        if A_true is not None:

            for metric_name in ["auc", "kendall_tau", "spearman_rho"]:
                results[metric_name] = eval_fns[metric_name](A_true, A_pred)

    # export evaluation results
    time = pd.Timestamp.now()
    df = pd.DataFrame(
        columns=[
            "timestamp",
            "dataset",
            "split_id",
            "model",
            "metric",
            "value",
            "config",
        ]
    )

    for metric_name, val in results.items():

        df.loc[len(df)] = (
            time,
            args.dataset,
            args.split_id,
            args.model,
            metric_name,
            val,
            vars(args),
        )

    logger.info(df)
    export_csv(df, osp.join(args.output_dir, "results.csv"), append=True)

