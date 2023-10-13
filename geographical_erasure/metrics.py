# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import scipy
import torch
from scipy.spatial import distance
from tqdm import tqdm


def compute_kl_divergence(p, q):
    """
    Computes the KL divergence of two distributions


    Parameters
    ----------
    p : dict, whose values represent the P distribution --> ground truth
    q : dict, whose values represent the Q distribution --> prediction

    Returns
    -------
    kl-divergence: float, kl between p and q
    """
    p = dict(sorted(p.items()))  # order alphabetically by key
    q = dict(sorted(q.items()))

    return sum(scipy.special.rel_entr(list(p.values()), list(q.values())))


def get_erasure_set(prediction, ground_truth, r):
    """
    Returns the erasure set according to Def. 1 and 2 in https://www.overleaf.com/project/640a065697c5029d320deae3,
    we return also the probabilities for each k, i.e. {k_1:p_1, ..., k_m:p_m} for m \in S_k.

    Parameters
    ----------
    prediction:  dict of input key, value pairs
    ground_truth : dict of input key, value pairs
    r: float, threshold for erasure definition

    Returns
    -------
    erasure_set: dict[float], dict {k_1:p_1, ..., k_m:p_m} for m \in S_k, the countries in the erasure set
    """
    assert math.isclose(
        np.sum(list(ground_truth.values())), 1
    ), "Ground truth is not a valid probability distribution."
    assert math.isclose(
        np.sum(list(prediction.values())), 1, abs_tol=1e-03
    ), f"Prediction is not a valid probability distribution as it sums to {np.sum(list(prediction.values()))}."
    assert r >= 0, "r smaller than 0 is not valid"

    erasure_set = {}
    for key in prediction.keys():
        if prediction[key] == 0:
            prediction[key] = 1e-7
        if ground_truth[key] / prediction[key] > r:
            erasure_set[key] = prediction[key]
    return erasure_set


def compute_metrics(all_predictions, ground_truth, use_erasure_set=False, r=3):
    """
    Returns metrics
    1) between individual prompts and ground truth
    2) the erasure sets for each prompt (if use_erasure_set=True, otherwise None)

    Parameters
    ----------
    all_predictions: list of dicts, predictions per prompt
    ground_truth : dict of input key, value pairs
    use_erasure_set : bool (opt), whether to compute kl on erasure set only (=ER^r), or on the full dataset
    r : float (opt), threshold r for erasure set definition

    Returns
    -------
    ground_truth_metrics: list[float], list of pairwise metrics (either kl or ER)
    erasure_sets: list[dict[float]], list of dict [{k_1:p_1, ..., k_m:p_m}_i] for m \in S_k, the countries in the erasure set for each prompt i
    """
    nr_prompts = len(all_predictions)
    erasure_sets = None

    if use_erasure_set:
        erasure_set_predictions = []
        erasure_sets = []
        for pred_i in all_predictions:
            erasure_set = get_erasure_set(pred_i, ground_truth, r=r)
            erasure_set_predictions.append(erasure_set)
            erasure_sets.append(erasure_set)
        all_predictions = erasure_set_predictions

    # evaluate metric between ground truth and individual prompts
    ground_truth_metrics = []
    for i in range(nr_prompts):
        ground_truth_on_Sr = {
            k: v for k, v in ground_truth.items() if k in all_predictions[i].keys()
        }
        ground_truth_metrics.append(
            compute_kl_divergence(ground_truth_on_Sr, all_predictions[i])
        )  # KL(P || Q)

    return ground_truth_metrics, erasure_sets


def perplexity(model, tokenizer, dataset, device):
    """
    Returns a `model`'s perplexity on a `dataset`.

    Parameters
    ----------
    model: A HF transformer model
    tokenizer: HF tokenizer
    dataset: HF dataset, we use wikitext-2-v1
    device: str, 'cpu' or 'gpu' depending on availability

    Returns
    -------
    ppl: perplexity of the `model` on the `dataset`
    """
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def compute_erasure_terms(predictions, ground_truth, r=3):
    """
    Returns individual erasure terms for each country, we use them for plotting (see teaser figure).

    Parameters
    ----------
    predictions: dict, predicted probabilities for each country
    ground_truth: dict, predicted probabilities for each country

    Returns
    -------
    erasure_terms: dict[float], ground_truth[i] * (log(predictions[i]) / log(ground_truth[i])) for i in S_r
    """
    erasure_terms = {}
    erasure_set = get_erasure_set(predictions, ground_truth, r=r)
    for country in predictions.keys():
        if country in erasure_set:
            p_tilde = ground_truth[country]
            p = predictions[country]
            erasure_terms[country] = p_tilde * (np.log(p_tilde) - np.log(p))
        else:
            erasure_terms[country] = 0
    return erasure_terms
