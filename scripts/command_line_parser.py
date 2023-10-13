# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse


def create_parser():
    """
    Creates argparser to handle command line arguments for experiment scripts.

    Returns
    -------
    parser: ArgumentParser containing the command line args
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        help="Which model to use (gpt2 / gpt2-medium / gpt2-large / gpt2-xl).",
    )
    parser.add_argument("--gpu", type=str, default="0", help="Which gpu to use.")
    parser.add_argument(
        "--learning_rate", type=str, default="3e-5", help="Which lr to use."
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="debugging",
        help="Where to save results.",
    )
    parser.add_argument(
        "--train_test_split_strategy",
        type=str,
        default="random",
        help="How to perform train test split.",
    )
    parser.add_argument(
        "--use_erasure",
        type=bool,
        default=False,
        help="Whether to use KL loss or EB^r loss for fine-tuning.",
    )
    parser.add_argument("--fold", type=int, default=0, help="Which fold.")
    parser.add_argument(
        "--r", type=float, default=3.0, help="Which r to use in the definition of ER^r."
    )
    return parser
