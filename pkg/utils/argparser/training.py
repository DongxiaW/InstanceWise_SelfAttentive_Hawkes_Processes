def add_base_arguments(parser):
    parser.add_argument(
        "--dataset", type=str, default="pgem-1K-5", help="default: pgem-1K-5"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/input",
        help="default: data/input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="default: data/output",
    )
    parser.add_argument("--split_id", type=int, default=0, help="default: 0")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")
    parser.add_argument("--cuda", action="store_true", help="default: false")
    parser.add_argument(
        "--skip_eval_infectivity", action="store_true", help="default: false"
    )
    parser.add_argument(
        "--skip_pred_next_event", action="store_true", help="default: false"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="default: false"
    )
    return parser


def add_subparser_arguments(model, subparsers):
    if model == "ISAHP":
        # add sub-parsers for each individual model
        sub_parser = subparsers.add_parser(
            model, help="Instancewise Self-Attentive Hawkes Processes"
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--embedding_dim", type=int, default=19, help="default: 19"
        )
        sub_parser.add_argument(
            "--hidden_size", type=int, default=20, help="default: 20"
        )
        sub_parser.add_argument(
            "--num_head", type=int, default=2, help="default: 2" #for alpha and gamma, each with 10 types
        )
        sub_parser.add_argument(
            "--batch_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--dropout", type=float, default=0.0, help="default: 0.0"
        )
        sub_parser.add_argument(
            "--lr", type=float, default=0.001, help="default: 0.001"
        )
        sub_parser.add_argument(
            "--epochs", type=int, default=200, help="default: 200"
        )
        sub_parser.add_argument(
            "--optimizer", type=str, default="Adam", help="default: Adam"
        )
        sub_parser.add_argument(
            "--l2_reg", type=float, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--l1_reg", type=float, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--type_reg", type=float, default=1, help="default: 1"
        )
        sub_parser.add_argument(
            "--num_workers", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--bucket_seqs",
            action="store_true",
            help="Whether to bucket sequences by lengths. default: False",
        )
        # for attributions
        sub_parser.add_argument(
            "--steps", type=int, default=50, help="default: 50"
        )
        sub_parser.add_argument(
            "--attr_batch_size", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--occurred_type_only",
            action="store_true",
            help="Whether to only use occurred event types in the batch as"
            "target types. default: False",
        )

        sub_parser.add_argument(
            "--tune_metric", type=str, default="nll", help="default: nll"
        )

    else:
        raise ValueError(f"model={model} is not supported.")

