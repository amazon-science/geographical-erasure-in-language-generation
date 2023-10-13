# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import command_line_parser
import numpy as np
import torch
from datasets import load_dataset
from finetuning_helpers import compute_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from geographical_erasure import data_utils
from geographical_erasure.distribution_model import DistributionModel
from geographical_erasure.experiment_utils import (
    load_ground_truth,
    load_model_and_tokenizer,
)
from geographical_erasure.loss import KLLoss
from geographical_erasure.metrics import perplexity
from geographical_erasure.prompt_dataset import PromptDataset, get_train_test_splits

nr_datapoints = 716
WARMUP_STEPS = nr_datapoints  # 5000
NR_EPOCHS = 5
NUM_TRAINING_STEPS = nr_datapoints * NR_EPOCHS
# one epoch = 955 prompts // steps --> start decaying after 3 epochs


def main(args):
    """Main method to run experiment.

    Parameters
    ----------
    args: argparse.ArgumentParser, containing experiment args
    """
    LEARNING_RATE = float(args.learning_rate)

    # set paths for saving results
    folder = f"../results/fine_tuning_results/{args.experiment_folder}"
    experiment_name = f"{folder}/{args.model_type}_{LEARNING_RATE}_split={args.train_test_split_strategy}_EBr={args.use_erasure}_fold={args.fold}_metrics.p"
    model_name = f"{folder}/models/{args.model_type}_{LEARNING_RATE}_split={args.train_test_split_strategy}_EBr={args.use_erasure}_fold={args.fold}_finetuned"

    if not os.path.exists(folder):
        # if it doesn't exist, create it
        os.makedirs(folder)

    if args.model_type != "gpt-neox-20b":
        # set gpu device if not gpu-parallel
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ground truth
    ground_truth_path = "../data/english-speaking-population-data.txt"
    ground_truth, synonym_dict = load_ground_truth(ground_truth_path, normalize=True)
    domain = list(ground_truth.keys())

    # init model
    base_model, tokenizer = load_model_and_tokenizer(args)
    distribution_model = DistributionModel(base_model, domain, device, synonym_dict)
    distribution_model = distribution_model.to(device)

    # init datasets
    prompt_path = "../data/population-prompts-automatic.txt"
    basic_prompts = data_utils.read_prompts(prompt_path)
    extended_prompts = data_utils.extend_prompt_list(basic_prompts)
    # run this twice in order to produce all combinations of pronouns + relations
    extended_prompts = data_utils.extend_prompt_list(extended_prompts)
    train_prompts, test_prompts = get_train_test_splits(
        extended_prompts, how=args.train_test_split_strategy, fold=args.fold
    )
    train_dataset = PromptDataset(
        train_prompts, ground_truth, tokenizer, synonym_dict=synonym_dict, device=device
    )
    test_dataset = PromptDataset(
        test_prompts, ground_truth, tokenizer, synonym_dict=synonym_dict, device=device
    )

    # set only biases trainable for bitfit strategy
    for name, param in distribution_model.named_parameters():
        param.requires_grad = False
        if "bias" in name:
            param.requires_grad = True

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )  # batch size 1 corresponds to one prompt
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )  # batch size 1 corresponds to one prompt

    # load wikitext to evaluate perplexity on
    perplexity_dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # init loss & optimizer
    kl_loss_function = KLLoss(
        ground_truth, device=device, use_erasure=args.use_erasure, r=3.0
    )
    optimizer = AdamW(distribution_model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_TRAINING_STEPS
    )

    # compute loss and perplexity once before fine-tuning
    distribution_model.eval()
    # train loss
    train_loss = compute_loss(distribution_model, train_loader, kl_loss_function)
    epoch_train_losses = [train_loss]
    print("Avg. train loss before fine-tuning:", train_loss)
    # test loss
    test_loss = compute_loss(distribution_model, test_loader, kl_loss_function)
    epoch_test_losses = [test_loss]
    print("Avg. test loss before fine-tuning:", test_loss)
    # perplexity
    pre_training_perplexity = perplexity(
        distribution_model.model, tokenizer, perplexity_dataset, device
    )
    print("Perplexity before fine-tuning:", pre_training_perplexity)
    epoch_perplexities = [pre_training_perplexity]

    # finetuning
    distribution_model.train()
    for epoch in range(NR_EPOCHS):
        print("\n Epoch :", epoch)
        for prompt in tqdm(train_loader):
            log_q = distribution_model(prompt, return_logprobs=True)
            loss = kl_loss_function(log_q)
            # monitor the loss ...
            # update the params
            loss.backward()
            optimizer.step()
            scheduler.step()
            # zero out the grads
            optimizer.zero_grad()
            distribution_model.zero_grad()

        # log train losses after the whole epoch is done to be consistent with test loss logging
        train_loss = compute_loss(distribution_model, train_loader, kl_loss_function)
        print("Avg. train loss:", train_loss)
        epoch_train_losses.append(train_loss)
        test_loss = compute_loss(distribution_model, test_loader, kl_loss_function)
        epoch_test_losses.append(test_loss)
        # perplexity
        epoch_perplexity = perplexity(
            distribution_model.model, tokenizer, perplexity_dataset, device
        )
        epoch_perplexities.append(epoch_perplexity)

    metrics_dict = {
        "train_kl_loss": epoch_train_losses,
        "test_kl_loss": epoch_test_losses,
        "perplexity": epoch_perplexities,
    }
    with open(experiment_name, "wb") as handle:
        pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save model for later
    distribution_model.model.save_pretrained(model_name)


if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)
