# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import geographical_erasure
from geographical_erasure import data_utils

sns.set_theme()

plt.rcParams.update({"font.size": 28})
plt.rc("axes", titlesize=24, labelsize=28)
plt.rc("ytick", labelsize=24)
plt.rc("xtick", labelsize=24)


def make_plot(model):
    """Generates plots from Fig. 2 and Fig. 7, comparing KL and ER^r for different choices of r.

    Parameters
    ----------
    model: HF model
    """
    ground_truth_file_geo = "../data/english-speaking-population-data.txt"
    ground_truth_geo = data_utils.read_population_data(ground_truth_file_geo)
    ground_truth_geo = geographical_erasure.normalize_dict(ground_truth_geo)

    # read list of basic prompts to extend automatically -- GEOGRAPHY
    basic_prompts_geo = data_utils.read_prompts(
        "../data/population-prompts-automatic.txt"
    )
    extended_prompts_geo = data_utils.extend_prompt_list(basic_prompts_geo)
    # run this twice in order to produce all combinations of pronouns + relations
    extended_prompts_geo = data_utils.extend_prompt_list(extended_prompts_geo)

    with open(
        f"../results_disambiguated/geography_prompt_predictions_{model}.p", "rb"
    ) as file:
        all_predictions_geo = pickle.load(file)  # already normalised

    # plotting
    # get kl divs
    r_levels = np.linspace(0, 10, 51)
    kl, pairwise_off_diags = geographical_erasure.compute_metrics(
        all_predictions_geo, ground_truth_geo
    )  # , use_erasure_set=False, r=10)
    median_kl = np.median(kl)
    perc25_kl = np.percentile(kl, 25)
    perc75_kl = np.percentile(kl, 75)
    # get Ers
    Er_per_level = []
    for r in tqdm(r_levels):
        Er_per_level.append(
            geographical_erasure.compute_metrics(
                all_predictions_geo, ground_truth_geo, use_erasure_set=True, r=r
            )[0]
        )
    median_Er_per_level = np.array(
        [np.median(prompt_kls) for prompt_kls in Er_per_level]
    )
    perc25_Er = np.percentile(Er_per_level, 25, axis=1)
    perc75_Er = np.percentile(Er_per_level, 75, axis=1)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    # ax.set_xscale('log')

    perc25_kl = np.percentile(kl, 25)
    perc75_kl = np.percentile(kl, 75)
    p1 = ax.plot(
        r_levels,
        [median_kl] * len(r_levels),
        c="darkred",
        label=r"KL$(\tilde{p} || p)$",
    )
    ax.fill_between(
        r_levels,
        perc25_kl,
        perc75_kl,
        alpha=0.4,
        edgecolor="darkred",
        facecolor="darkred",
    )

    perc25_Er = np.percentile(Er_per_level, 25, axis=1)
    perc75_Er = np.percentile(Er_per_level, 75, axis=1)
    p2 = ax.plot(
        r_levels, median_Er_per_level, c="darkblue", label=r"ER$^r(\tilde{p},  p)$"
    )
    ax.fill_between(
        r_levels,
        perc25_Er,
        perc75_Er,
        alpha=0.4,
        edgecolor="darkblue",
        facecolor="darkblue",
    )

    ax.set_xticks(r_levels[::5])
    # labels = [r'$10^0$', '', '', '', '', r'$10^1$', '', '', '', '',  r'$10^2$']
    # ax.set_xticklabels(labels)

    p3 = ax.axvline(3, c="k", linestyle="dashed", label="$r=3$")

    ax.set_xlabel(r"$r$")
    ax.legend(fontsize=20)  # , loc=4)
    plt.title(r"ER$^r$ and KL for " + str(model), fontsize=28)
    plt.axvspan(-0.2, 1, facecolor="0.2", alpha=0.3)  # , zorder=-100)
    plt.xlim([0, r_levels[-1]])
    plt.savefig(f"images/erasure_set_vs_kl_{model}.pdf")


def main():
    models = [
        "open_llama_3b",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "gpt-neo-125M",
        "gpt-neo-1.3B",
        "gpt-neo-2.7B",
        "gpt-neox-20b",
    ]
    for model in models:
        print(f"Processing {model}.")
        make_plot(model)


if __name__ == "__main__":
    main()
