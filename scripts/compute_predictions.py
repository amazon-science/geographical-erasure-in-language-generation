# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import command_line_parser
import pickle5 as pickle  # on ec2
import torch
from torch.utils.data import DataLoader

# import pickle # on mac
from tqdm import tqdm

import geographical_erasure
import geographical_erasure.data_utils as data_utils
from geographical_erasure.distribution_model import DistributionModel
from geographical_erasure.experiment_utils import (
    load_ground_truth,
    load_model_and_tokenizer,
)
from geographical_erasure.prompt_dataset import PromptDataset


def prepare_experiment_paths(args):
    """Gets paths to load data and save results.

    Parameters
    ----------
    args: argparse.ArgumentParser, containing experiment args

    Returns
    -------
    paths, tuple(str) of 5 paths
    """
    # construct experiment parameters (e.g. file paths)
    prompt_path = "../data/population-prompts-automatic.txt"
    ground_truth_path = "../data/english-speaking-population-data.txt"

    folder = f"../results/{args.experiment_folder}"
    if not os.path.exists(folder):
        # Create a new directory because it does not exist
        os.makedirs(folder)
    results_path_individual = (
        f"{folder}/geography_prompt_predictions_{args.model_type}.p"
    )
    results_path_aggregated = (
        f"{folder}/geography_aggregated_predictions_{args.model_type}.p"
    )
    results_path_normalising_constants = (
        f"{folder}/geography_normalising_constants_{args.model_type}.p"
    )
    return (
        prompt_path,
        ground_truth_path,
        results_path_individual,
        results_path_aggregated,
        results_path_normalising_constants,
    )


def main(args):
    """Main method that runs the experiment.

    Parameters
    ----------
    args: argparse.ArgumentParser, containing experiment args

    """
    if args.model_type != "gpt-neox-20b" and not "llama" in args.model_type:
        # set gpu device if not gpu-parallel
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get paths
    (
        prompt_path,
        ground_truth_path,
        results_path_individual,
        results_path_aggregated,
        results_path_normalising_constants,
    ) = prepare_experiment_paths(args)
    # init model
    ground_truth, synonym_dict = load_ground_truth(ground_truth_path)
    countries = list(ground_truth.keys())
    base_model, tokenizer = load_model_and_tokenizer(args)
    base_model.eval()
    distribution_model = DistributionModel(base_model, countries, device, synonym_dict)
    distribution_model = distribution_model.to(device)

    # init dataset
    basic_prompts = data_utils.read_prompts(prompt_path)
    extended_prompts = data_utils.extend_prompt_list(basic_prompts)
    # run this twice in order to produce all combinations of pronouns + relations
    extended_prompts = data_utils.extend_prompt_list(extended_prompts)

    prompt_dataset = PromptDataset(
        extended_prompts,
        ground_truth,
        tokenizer,
        synonym_dict=synonym_dict,
        device=device,
    )
    prompt_dataloader = DataLoader(
        prompt_dataset, batch_size=1, shuffle=False
    )  # batch size 1 corresponds to one prompt

    print("Iterating over all prompts to make predictions.")
    all_predictions = []
    normalisation_constants = []
    with torch.no_grad():
        for prompt in tqdm(prompt_dataloader):
            probs = distribution_model(prompt, return_logprobs=False)
            probs = {countries[i]: probs[i].cpu().item() for i in range(len(countries))}
            all_predictions.append(probs)

    print('Aggregating individual predictions to compute "marginal".')
    aggregated_probs = geographical_erasure.get_aggregated_likelihoods(
        base_model,
        tokenizer,
        extended_prompts,
        ground_truth,
        precomputed_probs=all_predictions,
    )

    print("Done, pickleing results.")
    with open(results_path_individual, "wb") as file:
        pickle.dump(all_predictions, file)
    with open(results_path_aggregated, "wb") as file:
        pickle.dump(aggregated_probs, file)
    with open(results_path_normalising_constants, "wb") as file:
        pickle.dump(normalisation_constants, file)


if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)
