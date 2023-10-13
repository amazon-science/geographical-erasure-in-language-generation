# ideas:
# normalization function: test that values indeed sum to 1 after normalization
from numpy import testing

from geographical_erasure import data_utils
from geographical_erasure.probe_lm import normalize_dict


def test_normalization():
    # test that the normalisation function indeed yields a distribution, i.e. the values sum to 1
    data_dict = {"a": 0.8, "b": 0.7, "c": 1.5}
    data_dict_normalised = normalize_dict(data_dict)
    testing.assert_equal(sum(list(data_dict_normalised.values())), 1)


def test_extend_prompts_order():
    # test that this function is deterministic,
    # specifically that the use of sets doesn't screw up the order:
    # the order should be the same when we run this function multiple times
    basic_prompts = data_utils.read_prompts("../data/population-prompts-automatic.txt")
    extended_prompts1 = data_utils.extend_prompt_list(basic_prompts)
    extended_prompts2 = data_utils.extend_prompt_list(basic_prompts)
    testing.assert_equal(extended_prompts1, extended_prompts2)
