# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import transformer libraries
import torch
from transformers import (  # gpt2; EleutherAI gpt-neo; open llama
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from geographical_erasure import data_utils, experiment_utils


def load_model_and_tokenizer(args):
    """
    Loads a pretrained model and a tokenizer from HF.

    Parameters
    ----------
    args: argparse namespace object containing model/experiment info. Importantly for this function, the `model_type` to use.

    Returns
    -------
    model, tokenizer
    """
    # load a huggingface model & tokenizer
    if "neox" in args.model_type:
        # neox needs to be run accross multiple gpus
        model = AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/gpt-neox-20b", device_map="auto", load_in_8bit=True
        )
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            f"EleutherAI/{args.model_type}"
        )
    elif "neo" in args.model_type:
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{args.model_type}")
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/{args.model_type}")
    elif "llama" in args.model_type:
        model_path = f"openlm-research/{args.model_type}"
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
    else:  # gpt2 models
        model = GPT2LMHeadModel.from_pretrained(
            args.model_type, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0
        )  # kwargs to make the model behave deterministically for fine-tuning experiment
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    return model, tokenizer


def load_ground_truth(
    ground_truth_path, normalize=False, synonym_dict_path="../data/country_synonyms.p"
):
    """
    Loads ground truth dict from file, and synonym dict if desired.

    Parameters
    ----------
    ground_truth_path: str, path to ground truth dict
    normalize: bool, whether to normalise population numbers, i.e.,
            return percentages instead of counts
    synonym_dict_path: str, path to synonym dict containing alternative country names


    Returns
    -------
    ground_truth: dict[float], dict containing (normalized) population counts
    synonym_dict: dict[list[str]], dict containing alternative country names
    """
    ground_truth, synonym_dict = data_utils.read_population_data(
        ground_truth_path, return_synonym_dict=True, synonym_dict_path=synonym_dict_path
    )
    if normalize:
        ground_truth = experiment_utils.normalize_dict(ground_truth)
    return ground_truth, synonym_dict


def normalize(vals, logscale=False):
    """
    Returns a normalized list of values (a[i] = a[i]/sum(a)).

    Parameters
    ----------
    vals: list of input values

    Returns
    -------
    (log_)vals: list[float], normalized list such that values add to 1 (or 0 if `logscale`=True)
    """
    if logscale:
        log_vals = torch.nn.functional.log_softmax(vals, dim=-1)
        return log_vals
    else:
        s = vals.sum()
        vals = vals / s
        return vals


def normalize_dict(data, return_sum=False):
    """
    Returns a normalized dictionary.

    Parameters
    ----------
    data : dict of input key, value pairs
    return_sum: whether to return the total mass assigned to all countries to get an idea how drastic the renormalisation was (default: False)

    Returns
    -------
    normalized_data : dict of input key, value pairs, now such that the values add to 1
    """
    normalized_data = data.copy()
    s = sum(data.values())
    if s == 0:
        return data

    for key in data:
        normalized_data[key] = float(data[key]) / s

    if return_sum:
        return normalized_data, s

    else:
        return normalized_data


def expand_candidates(domain, synonym_dict=None):
    """
    Expands list of candidates by appending all synonyms. The order is all original countries then all synonyms (in the same order).

    Parameters
    ----------
    domain: list[str], original country names
    synonym_dict: dict[list[str]], list of synonyms

    Returns
    -------
    candidates: list[str], flat list of extended country names (original country names + their synonyms)
    """
    if synonym_dict is not None:
        all_countries = [*synonym_dict.keys()]
        assert domain == all_countries, "Data and synonyms dict do not match!"
        for c in synonym_dict.keys():
            for alt_c in synonym_dict[c]:
                all_countries.append(alt_c)
        candidates = all_countries
    else:
        candidates = domain
    return candidates
