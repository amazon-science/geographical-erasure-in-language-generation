import numpy as np
import pytest
import torch
from numpy import testing
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import geographical_erasure
from geographical_erasure.distribution_model import DistributionModel
from geographical_erasure.experiment_utils import load_ground_truth
from geographical_erasure.loss import KLLoss
from geographical_erasure.metrics import compute_kl_divergence
from geographical_erasure.prompt_dataset import PromptDataset


# Test that batched results using lightning-style modules are the same as using for-loops
def test_no_agg():
    # load a huggingface model
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ground_truth, _ = load_ground_truth(
        ground_truth_path="../data/english-speaking-population-data.txt",
        normalize=True,
        synonym_dict_path="../data/country_synonyms.p",
    )
    domain = [*ground_truth.keys()]
    prompt = "I live in"

    # for-loop forward pass
    population_probs_for_loop = geographical_erasure.compute_population_probs(
        gpt2_model,
        gpt2_tokenizer,
        prompt,
        {country: 1.0 / len(domain) for country in domain},
        normalise=True,
    )
    for_loop_values = [*population_probs_for_loop.values()]

    # batched forward pass
    dataset = PromptDataset([prompt], ground_truth, gpt2_tokenizer, synonym_dict=None)
    model = DistributionModel(gpt2_model, domain, device="cpu")
    for x in dataset:  # y is a dummy
        batched_values = model(x, return_logprobs=False).detach()
        break
    testing.assert_almost_equal(for_loop_values, batched_values, decimal=5)


@pytest.mark.parametrize("log_scale", [True, False])
def test_aggregation(log_scale):
    # load a huggingface model
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    ground_truth_path = "../data/english-speaking-population-data.txt"

    ground_truth, synonym_dict = load_ground_truth(
        ground_truth_path,
        normalize=True,
        synonym_dict_path="../data/country_synonyms.p",
    )
    domain = list(ground_truth.keys())
    prompt = "I live in"

    # old forward pass
    population_probs_for_loop = geographical_erasure.compute_population_probs(
        gpt2_model,
        gpt2_tokenizer,
        prompt,
        ground_truth,
        normalise=True,
        synonym_dict=synonym_dict,
    )  # Attention! not on log scale
    for_loop_values = [*population_probs_for_loop.values()]
    if log_scale:
        for_loop_values = [np.log(p) for p in for_loop_values]

    # batched forward pass
    dataset = PromptDataset(
        [prompt], ground_truth, gpt2_tokenizer, synonym_dict=synonym_dict
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None
    )  # now automatically includes all candidates
    # model
    model = DistributionModel(
        gpt2_model, domain, device="cpu", synonym_dict=synonym_dict
    )
    for x in dataloader:  # y is a dummy
        batched_output = model(
            x, return_logprobs=log_scale
        )  # one batch cointains 251 sentences
        # batched_output shape: [countries + synonyms, 1]

    batched_result = batched_output.detach().numpy()
    testing.assert_almost_equal(for_loop_values, batched_result, decimal=3)

    if log_scale:
        # test loss function
        loss_module = KLLoss(ground_truth, device="cpu", use_erasure=False)
        kl_loss_module = loss_module(batched_output).detach().cpu().numpy()
        for_loop_loss = compute_kl_divergence(ground_truth, population_probs_for_loop)
        testing.assert_almost_equal(for_loop_loss, kl_loss_module, decimal=3)
