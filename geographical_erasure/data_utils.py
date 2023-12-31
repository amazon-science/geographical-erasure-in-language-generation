# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pickle5 as pickle  # on ec2

# import pickle # on mac


def remove_spl_characters(word):
    """
    Removes special symbols used to denote space (Ġ) and newline (Ċ).

    Parameters
    ----------
    word : str, the input word

    Returns
    -------
    word: str, word without special characters
    """
    word = word.replace("Ġ", "")
    word = word.replace("Ċ", "")
    return word


def read_population_data(
    filename="data/world-population.txt",
    return_synonym_dict=False,
    synonym_dict_path=None,
):
    """
    Reads the population stats from the filename and
    returns a dictionary {country1: population1, ...}.

    Parameters
    ----------
    filename : str, the input filename
    return_synonym_dict : bool, whether or not to return also the dict of synonyms
    synonym_dict_path: str, if return_synonym_dict is True one must pass a path to load from here

    Returns
    -------
    population_stats: dict, dictionary with {country1: population1, ...}
    synonym_dict (optional): dict[list[str]], dictonary containing list of synonyms for each country,
        returned only if `synonym_dict_path` is not None

    """
    f = open(filename, "r")
    lines = f.readlines()
    population_stats = {}
    for line in lines:
        country = line.strip().split("\t")[0]
        population = line.strip().split("\t")[1]
        population_stats[country] = int(population.replace(",", ""))

    if return_synonym_dict:
        assert (
            synonym_dict_path is not None
        ), "To `return_synonym_dict` you must pass a file path."
        with open(synonym_dict_path, "rb") as handle:
            synonym_dict = pickle.load(handle)
        return population_stats, synonym_dict
    else:
        return population_stats


def read_prompts(filename):
    """
    Reads the prompts from the given filename.

    Parameters
    ----------
    filename : str, the input filename

    Returns
    -------
    lines: list[str], list of prompts
    """

    f = open(filename, "r")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def exchange_pronouns(prompt, pronoun_list):
    """
    Helper for generating more diverse prompts: exchange pronous.

    Parameters
    ----------
    prompt: prompt in which to exhange pronouns
    pronoun_list: list of pronouns that can be replaced with each other to form a grammatical sentence

    Returns
    -------
    additional_promots: list[str], list of additional prompts that have been generated by exchanging pronouns
    """
    additional_prompts = []
    word_to_replace = list(set(prompt.split(" ")).intersection(pronoun_list))
    if word_to_replace:  # if non empty set
        word_to_replace = word_to_replace[0]
        for pronoun in pronoun_list:
            new_prompt = prompt.replace(word_to_replace, pronoun)
            additional_prompts.append(new_prompt)
    return additional_prompts


def get_replacer_lists():
    """
    Helper for generating more diverse prompts: list all words to be exchnaged with each other

    Parameters
    ----------
    None

    Returns
    -------
    replacer_lists: list[list[str]], list of additional prompts that have been generated by exchanging pronouns
    """
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

    replacer_lists = [
        third_singular_pronouns,
        plural_pronouns,
        possessive_pronouns,
        person_relations,
    ]
    return replacer_lists


def extend_prompt_list(basic_prompts):
    """
    Iterate through list of prompts to make it more diverse by replacing pronouns and relations.

    Parameters
    ----------
    basic_prompts: list of prompts to extend via exchanging

    Returns
    -------
    extended_prompts: list of extended prompts (original + new)
    """
    replacer_lists = get_replacer_lists()

    extended_prompts = basic_prompts.copy()
    for prompt in basic_prompts:
        for pronoun_list in replacer_lists:
            extended_prompts += exchange_pronouns(prompt, pronoun_list)
    extended_prompts = list(set(extended_prompts))

    remove_prompts = [
        "We are a citizen of"
    ]  # not grammatically correct prompts that we generate accidentally
    for remove_p in remove_prompts:
        if remove_p in extended_prompts:
            extended_prompts.remove(remove_p)
    extended_prompts = sorted(extended_prompts)
    return extended_prompts
