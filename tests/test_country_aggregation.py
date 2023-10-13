import numpy as np
from numpy import testing
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from geographical_erasure import compute_population_probs
from geographical_erasure.data_utils import read_population_data


def test_loading():
    population_data = read_population_data(
        filename="../data/english-speaking-population-data.txt"
    )
    population_data, synonym_dict = read_population_data(
        filename="../data/english-speaking-population-data.txt",
        return_synonym_dict=True,
        synonym_dict_path="../data/country_synonyms.p",
    )
    assert population_data.keys() == synonym_dict.keys()
    # test that we raise an error when return_synonym_dict=True but no path is given
    testing.assert_raises(
        AssertionError,
        read_population_data,
        filename="../data/english-speaking-population-data.txt",
        return_synonym_dict=True,
    )


def test_predictions():
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "I live in"

    population_data, synonym_dict = read_population_data(
        filename="../data/english-speaking-population-data.txt",
        return_synonym_dict=True,
        synonym_dict_path="../data/country_synonyms.p",
    )

    predictions = compute_population_probs(
        gpt2_model, gpt2_tokenizer, prompt, population_data, normalise=False
    )

    # when there are no synonyns nothing should change through the aggregation
    dummy_synonyms_dict = {country: [] for country in population_data.keys()}
    predictions_disambiguated_dummy = compute_population_probs(
        gpt2_model,
        gpt2_tokenizer,
        prompt,
        population_data,
        synonym_dict=dummy_synonyms_dict,
        normalise=False,
    )
    assert predictions == predictions_disambiguated_dummy

    # when there are synonyms the aggregated probabilities should be larger than the non-aggregated ones (if we don't normalise at least)
    predictions_disambiguated = compute_population_probs(
        gpt2_model,
        gpt2_tokenizer,
        prompt,
        population_data,
        synonym_dict=synonym_dict,
        normalise=False,
    )
    assert (
        np.array([*predictions_disambiguated.values()])
        >= np.array([*predictions.values()])
    ).all()
