from numpy import testing
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import geographical_erasure
from geographical_erasure import data_utils, get_aggregated_likelihoods
from geographical_erasure.experiment_utils import normalize_dict


# `agg_likelihoods` now supports pre-computed model prediction probabilities. This test checks that the function returns the
# same results when using precomputed probs and when running from scratch.
def test_agg_likelihoods_precomputed():
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    example_prompts = ["I live in", "You live in"]

    ground_truth_file = "../data/english-speaking-population-data.txt"
    ground_truth = data_utils.read_population_data(ground_truth_file)

    # 1. run without precomputed probabilities
    probs_from_scratch = geographical_erasure.get_aggregated_likelihoods(
        gpt2_model,
        gpt2_tokenizer,
        example_prompts,
        ground_truth,
        precomputed_probs=None,
    )

    # 2. run with precomputed probabilities
    all_predictions = []
    for prompt in tqdm(example_prompts):
        gpt2_probs = geographical_erasure.compute_population_probs(
            gpt2_model, gpt2_tokenizer, [prompt], ground_truth
        )
        normalized_gpt2_probs = normalize_dict(gpt2_probs)
        all_predictions.append(normalized_gpt2_probs)

    probs_precomputed = geographical_erasure.get_aggregated_likelihoods(
        gpt2_model,
        gpt2_tokenizer,
        example_prompts,
        ground_truth,
        precomputed_probs=all_predictions,
    )

    # 3. check that results are the same
    testing.assert_almost_equal(
        list(probs_from_scratch.values()), list(probs_precomputed.values())
    )


def test_agg_likelihoods_is_deterministic():
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    example_prompts = ["I live in", "You live in"]

    ground_truth_file = "../data/english-speaking-population-data.txt"
    ground_truth = data_utils.read_population_data(ground_truth_file)

    # precompute probabilities
    all_predictions = []
    for prompt in tqdm(example_prompts):
        gpt2_probs = geographical_erasure.compute_population_probs(
            gpt2_model, gpt2_tokenizer, [prompt], ground_truth
        )
        normalized_gpt2_probs = normalize_dict(gpt2_probs)
        all_predictions.append(normalized_gpt2_probs)

    US_mass = []
    for trial in range(10):
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        trial_probabilities = geographical_erasure.get_aggregated_likelihoods(
            gpt2_model,
            gpt2_tokenizer,
            example_prompts,
            ground_truth,
            precomputed_probs=all_predictions,
        )
        US_mass.append(trial_probabilities["the United States"])


if __name__ == "__main__":
    test_agg_likelihoods_is_deterministic()
