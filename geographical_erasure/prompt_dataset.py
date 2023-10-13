# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from geographical_erasure.experiment_utils import expand_candidates


class PromptDataset(Dataset):
    """
    Torch dataset to hold the prompts. It tokenizes all prompts + candidate countries on initialisation.
    Batching is handled here, i.e. tensors corresponding to the tokenised versions of one prompt + _all_ candidates
    countries are returned by `__getitem__`.

    Attributes
    ----------
    prompts: List[str], all prompts ('I live in', 'he is a citizen of', ...)
    domain: List[str], list of countries for which to make predictions
    tokenizer: HF tokenizer
    data: Tensor, contains tokenized prompts + candidate countries, shape is [#prompts, #countries]
    prompt_length: length of each prompt, needed to read out logits for conditional probabilities

    Methods
    -------
    __len__(): returns nr of prompts (here equal to number of batches)
    __getitem__(idx): returns a single batch, corresponding to tokenizer([prompt + country for country in domain])
    """

    def __init__(
        self, prompts, ground_truth, tokenizer, synonym_dict=None, device="cpu"
    ):
        self.prompts = prompts
        domain = [*ground_truth.keys()]
        self.domain = domain
        if synonym_dict is not None:
            candidates = expand_candidates(
                domain, synonym_dict=synonym_dict
            )  # model(x, y) shape: [len(candidates), len(prompt)]
        else:
            candidates = domain

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # only for gpt2-style models
        self.data = [
            self.tokenizer(
                [prompt + " " + candidate for candidate in candidates],
                return_tensors="pt",
                padding="max_length",
                max_length=42,
            ).to(device)
            for prompt in self.prompts
        ]  # data is the prompts + candidates

        self.prompt_length = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.prompt_length = len(self.tokenizer(self.prompts[idx])["input_ids"])
        encoded_batch = self.data[idx]
        # print(encoded_batch["attention_mask"].shape)
        # print(torch.zeros(encoded_batch["attention_mask"].shape[0], dtype=torch.int32).shape)
        return {
            "prompt_length": torch.LongTensor([self.prompt_length]),
            "attention_mask": encoded_batch["attention_mask"],
            "input_ids": encoded_batch["input_ids"],
        }


# helper functions to generate the train-test splits
def includes_word_from_list(prompt, word_list):
    """Checks whether a prompt contains any of the words from the list

    Parameters
    ----------
    prompt : str
        string to check
    word_list : List[str]
        list of words to check against

    Returns
    -------
    bool:
        whether the prompt contains any of the words from the `word_list`
    """
    is_included = False
    for word in word_list:
        if word in prompt:
            is_included = True
    return is_included


# word lists for splits below
third_singular_pronouns = ["She", "He"]
plural_pronouns = ["We", "They", "You"]
possessive_pronouns = ["My", "Her", "His", "Our", "Their"]
person_relations = [
    "uncle",
    "aunt",
    "uncle",
    "brother",
    "sister",
    "niece",
    "nephew",
    "mother",
    "father",
    "mom",
    "daughter",
    "son",
    "cousin",
    "friend",
    "relative",
]
pronouns = (
    third_singular_pronouns + plural_pronouns + possessive_pronouns + person_relations
)
verbs = [
    "hail",
    "homeland",
    "come",
    "citizen",
    "originate",
    "roots are",
    "grew up in",
    "brought up in",
    "raised in",
    "born in",
    "place of origin is",
    "live",
    "reside",
    "home country is",
]


def get_train_test_splits(prompts, split_size=0.75, how="random", fold=0):
    """Returns train test splits for finetuning experiment, according to the three strategies.

    Parameters
    ----------
    prompts: List[str],
        list of all prompts
    split_size: float, optional
        which proportion of prompts should be in the training set, by default 0.75
    how: str, optional
        strategy by which to split, by default 'random', other choices are 'pronouns' and 'verbs'
    fold: int, optional
        for running multiple folds and controlling stochasticity, by default 0

    Returns
    -------
    Tuple(List(str), List(str))
        train and test sets of prompts
    """
    assert how in [
        "random",
        "pronouns",
        "verbs",
    ], "Choose a valid strategy for `how`: ['random', 'pronouns', 'verbs']"
    random.seed(fold)
    if how == "random":
        train = random.sample(prompts, int(len(prompts) * split_size))
    if how == "pronouns":
        train_pronouns = random.sample(
            pronouns, int(len(pronouns) * (split_size - 0.25))
        )  # manually adjusted split size so that absolute sizes match
        train = [
            prompt
            for prompt in prompts
            if includes_word_from_list(prompt, train_pronouns)
        ]
    if how == "verbs":
        train_verbs = random.sample(verbs, int(len(verbs) * (split_size + 0.1)))
        train = [
            prompt for prompt in prompts if includes_word_from_list(prompt, train_verbs)
        ]
    test = list(set(prompts).difference(set(train)))
    return train, test
