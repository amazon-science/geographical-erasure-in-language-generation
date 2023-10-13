# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_scatter

from geographical_erasure.experiment_utils import normalize


class DistributionModel(nn.Module):
    """Torch model that receives a batch of tokenized prompt + countries and computes p(country | prompt) for all counties in a batch in the forward pass.
    It also handles the aggregation of synonym countries.


    Attributes
    ----------
    model: A HF transformer model used to compute the individual probabilities.
    synonym_dict: dict(str), dict to aggregate country preditions for alternative names
    country_mapping: a list which maps synonyms back to their canonical names, based on indices.
                For example, the ['UK', 'Italy', 'United Kingdom'] --> [0, 1, 0]

    Methods
    -------
    aggregate_synonyms(x, log_scale): aggregate the country predictions based on self.country_mapping. For the above example:
            [0.5, 0.3, 0.2] --> [0.7, 0.3], since predictions for 'UK' and 'United Kingdom' are added.
            `log_scale` indicates whether we aggregate p's, in which case we add, or log(p)'s, in which case we aggregate
            via logsumexp.
    forward(x, return_logprobs):
        Compute the forward pass. x is a dict with keys "prompt_length", "attention_mask" and "input_ids" as returned by the tokenizer.
    """

    def __init__(self, pretrained_model, domain, device, synonym_dict=None):
        super().__init__()
        self.model = pretrained_model
        self.synonym_dict = synonym_dict
        if self.synonym_dict is not None:
            self.country_mapping = list(
                range(len(domain))
            )  # each country once to start with
            for i, c_name in enumerate(domain):
                # order: all countries once, then all synonyms
                self.country_mapping += [
                    i for _ in range(len(self.synonym_dict[c_name]))
                ]  # append synonyms if any
            self.country_mapping = (
                torch.tensor(self.country_mapping).flatten().to(device=device)
            )

    def aggregate_synonyms(self, x, log_scale=True):
        if log_scale:
            aggregated_input = torch_scatter.scatter_logsumexp(
                x, self.country_mapping, 0
            )
        else:
            aggregated_input = torch.zeros(
                max(self.country_mapping) + 1, dtype=torch.float32, device=x.device
            )
            # not on log scale
            aggregated_input = aggregated_input.index_add(0, self.country_mapping, x)
        return aggregated_input

    def forward(self, x, return_logprobs=True):
        # manually fix the batch dimensions
        if len(x["input_ids"].shape) == 3:
            x["input_ids"] = x["input_ids"].squeeze(0)
            x["attention_mask"] = x["attention_mask"].squeeze(0)

        prompt_length = x["prompt_length"][0].item()
        outputs = self.model(
            **{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]},
            labels=x["input_ids"]
        )

        # get logits for candidates, i.e. after the prompt length
        logits = outputs.logits[
            :, prompt_length - 1 : -1, :
        ]  # offset by 1 bc logits starts with p(x_1 | x_0)

        mask = x["attention_mask"][:, prompt_length:].bool()
        logits[~mask] = float("-inf")

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        candidate_log_probs = torch.gather(
            log_probs, 2, x["input_ids"][:, prompt_length:][..., None]
        ).squeeze(-1)

        candidate_log_probs[~mask] = 0.0
        candidate_log_probs = candidate_log_probs.sum(-1)

        if return_logprobs:
            probs = candidate_log_probs
        else:
            probs = torch.exp(candidate_log_probs)

        if self.synonym_dict is not None:
            probs = self.aggregate_synonyms(probs, log_scale=return_logprobs)

        probs = normalize(probs, logscale=return_logprobs)
        return probs
