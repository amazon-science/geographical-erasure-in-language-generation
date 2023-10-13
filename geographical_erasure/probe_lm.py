# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast  # EleutherAI gpt-neo
from transformers import GPT2Tokenizer, GPT2TokenizerFast  # gpt2

from geographical_erasure.data_utils import remove_spl_characters
from geographical_erasure.experiment_utils import normalize_dict


def next_few_words(model, tokenizer, prompt, k=15):
    """
    Returns the most likely next words with their probability.

    Parameters
    ----------
    model: A transformer model
    tokenizer:
    prompt: str, a prefix containing the left context
    k: int, optional ... the value of top-k

    Returns
    -------
    list((str, float)), list of top_k next words and their probabilities
    """
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])

    # get logits
    logits = outputs.logits
    # get logits for the next word
    next_word_logits = logits[0, -1, :]
    # get probs for the next word
    next_word_probs = torch.nn.functional.softmax(next_word_logits, dim=-1)

    # sort the logits to give you the top ranking indices
    top_word_indices = torch.argsort(next_word_logits, descending=True)
    # convert the top indices into tokens
    top_k_words = tokenizer.convert_ids_to_tokens(top_word_indices[:k])
    top_k_probs = next_word_probs[top_word_indices[:k]]

    # clean the tokens
    top_k_words = [remove_spl_characters(word) for word in top_k_words]

    return [(word, prob.item()) for word, prob in zip(top_k_words, top_k_probs)]


def get_aggregated_likelihoods(
    model,
    tokenizer,
    list_of_prompts,
    data,
    precomputed_probs=None,
    return_prompt_probs=False,
):
    """
    Aggregates the likelihoods across a list of (diverse) prompts.

    Compute: \sum_{prompts} P(prompt) x P(candidate | prompt)

    Parameters
    ----------
    model: A transformer model
    tokenizer: a tokenizer
    list_of_prompts: list, a list of strings containing diverse prompts
    data: dict, a dictionary where keys represent the candidates
    precomputed_probs: list of dict, optional input of precomputed prompt distributions of length len(list_of_prompts);
        precomputed_probs[i] is the {candidate_k: p(candidate_k)}_k dict returned by `compute_population_probs`,
        precomputed_probs dicts must be normalised, otherwise we throw an error
    return_prompt_probs: whether to return the relative probability of each prompt along with the avg. distribution, default: False

    Returns
    -------
    likelihoods: dict[float], dict with the same keys as data and values are aggregated likelihoods
    """
    if precomputed_probs is not None:
        assert len(precomputed_probs) == len(
            list_of_prompts
        ), "len(precomputed_probs) must be equal to len(list_of_prompts)"
        for i in range(len(precomputed_probs)):
            assert math.isclose(
                sum(list(precomputed_probs[i].values())), 1, abs_tol=1e-3
            ), f"precomputed_probs must be normalised but sum to: {sum(list(precomputed_probs[i].values()))})"

    # initialize the likelihoods for all the candidates
    likelihoods = {}
    for key in data:
        likelihoods[key] = 0.0
    prompt_probs = {}

    for i, prompt in enumerate(list_of_prompts):
        # compute likelihoods for the given prompt
        if precomputed_probs is None:
            individual_likelihoods = compute_population_probs(
                model, tokenizer, prompt, data
            )
        else:
            individual_likelihoods = precomputed_probs[i]

        # compute prob (prompt)
        prob_prompt = probability(model, tokenizer, prompt)
        prompt_probs[prompt] = prob_prompt
        # compute joint likelihoods: P(prompt, candidate) = P(candidate | prompt) x P(prompt);
        # --> note that P(prompt) is not normalised, we do this below using the normalise function

        for key in likelihoods:
            likelihoods[key] += prob_prompt * individual_likelihoods[key]

    prompt_probs = normalize_dict(prompt_probs)
    likelihoods = normalize_dict(likelihoods)

    if return_prompt_probs:
        return likelihoods, prompt_probs

    return likelihoods


def probability(model, tokenizer, sentence):
    """
    Computes P(sentence).

    Parameters
    ----------
    model: A HF transformer model
    tokenizer: HF tokenizer.
    sentence: str, sentence for which we want to compute probability

    Returns
    -------
    p(sentence): float
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    return math.exp(-1 * loss.item())


def prob_given_context(model, tokenizer, prompt, candidate):
    """
    Computes p(candidate | prompt).

    Parameters
    ----------
    model: A transformer model
    tokenizer:
    prompt: str, a prefix containing the left context
    candidate: str, possible completion

    Returns:
    probs: float, p(candidate | prompt).
    """
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_prefix_space=False)

    # gpt neox tokenizer needs different inputs so we determine the type first
    if isinstance(tokenizer, GPTNeoXTokenizerFast) or isinstance(
        tokenizer, GPT2TokenizerFast
    ):
        candidate_inputs = tokenizer(candidate, return_tensors="pt")
    elif isinstance(tokenizer, GPT2Tokenizer):
        candidate_inputs = tokenizer(
            candidate, return_tensors="pt", add_prefix_space=True
        )
    else:
        raise ValueError("Tokenizer not supported.")

    # candidates
    candidate_ids = candidate_inputs["input_ids"]

    # compute next probs ...
    log_probs = 0.0

    for candidate_idx in candidate_ids[0]:
        candidate_idx = candidate_idx.item()

        # forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])

        # get logits
        logits = outputs.logits  # * torch.exp(-log_tau)
        # get logits for the next word
        next_word_logits = logits[
            0, -1, :
        ]  # outputs.logits.shape torch.Size([1, 3, 50257])  (for “I live in”)

        # get probs for the next word
        next_word_log_probs = torch.nn.functional.log_softmax(next_word_logits, dim=-1)

        log_probs += next_word_log_probs[candidate_idx]  # will this lead to underflow??

        # update the next_word_log_prob
        new_candidate_tensor = torch.tensor([[candidate_idx]], dtype=torch.int64)
        inputs["input_ids"] = torch.cat(
            (inputs["input_ids"], new_candidate_tensor), dim=1
        )  # add the first part of the candidate to the prompt
        inputs["attention_mask"] = torch.ones_like(
            inputs["input_ids"], dtype=torch.int64
        )  # attention on everything

    # convert to normal prob
    probs = math.exp(log_probs)
    return probs


def compute_population_probs(
    model, tokenizer, prompt, ground_truth, synonym_dict=None, normalise=True
):
    """
    Computes the likelihood for all the keys in the data for a given prompt
    For example, for a given prompt (e.g., I live in __), it computes
    the likelihood for each candidate (e.g., each country). Returns normalised population probabilities.

    Parameters
    ----------
    model: pretrained HF model
    tokenizer: pretrained HF tokenizer
    prompt: str, prompt for which to query the model, i.e. 'I live in'
    ground_truth: dict, keys of this dict are the countries for which we predict
    synonym_dict : dict, optional, pass dict of synonyms to disambiguate countries with

    Returns
    -------
    population_output: dict[float], {country: p(country | prompt)} for all countries in `ground_truth.keys()`
    """
    # if disambiguating: expand rephrased list into individual candidates
    if synonym_dict is not None:
        assert (
            ground_truth.keys() == synonym_dict.keys()
        ), "Data and synonyms dict do not match!"
        all_countries = [*synonym_dict.keys()]
        for c in synonym_dict.keys():
            for alt_c in synonym_dict[c]:
                all_countries.append(alt_c)
        candidates = all_countries
    else:
        candidates = [*ground_truth.keys()]

    output = {}
    for candidate in tqdm(candidates):
        prob = prob_given_context(model, tokenizer, prompt, candidate)
        output[candidate] = prob

    # if disambiguating: use rephrased list to aggregate countries
    # start  with the original country prob.
    population_output = {country: output[country] for country in ground_truth.keys()}

    if synonym_dict is not None:
        for country in ground_truth.keys():
            for synonym in synonym_dict[country]:
                population_output[country] += output[synonym]

    if normalise:  # if False we return raw probs
        population_output = normalize_dict(population_output)

    return population_output


def normalize(vals):
    """
    Normalise a list of values.

    Parameters
    ----------
    vals : list of input values

    Returns:
    --------
    list[str], a normalized list of values (a[i] = a[i]/sum(a))
    """
    s = sum(vals)
    if s == 0:
        return vals
    return [float(i) / s for i in vals]
