# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class KLLoss(nn.Module):
    """Torch loss module that computes KL loss between p(country | prompt) distribution and ground truth.

    Attributes
    ----------
    ground_truth: tensor, ground truth values for all countries
    use_erasure: bool, whether to compute erasure loss or full KL
    r: float, r value for erasure loss
    kl_loss: nn.Module, standard torch KL loss which is used under the hood


    Methods
    -------
    forward: forward pass computing the kl loss for (aggregated) predictions.
    """

    def __init__(self, ground_truth, device="cpu", use_erasure=False, r=3.0):
        super().__init__()
        self.ground_truth = torch.tensor([*ground_truth.values()]).to(device)
        self.use_erasure = use_erasure
        self.r = torch.tensor(r, requires_grad=False)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(self, x):  # no y since ground truth is constant & saved as part of loss
        x = x.reshape(-1)
        if self.use_erasure:
            erasure_indices = torch.where(
                self.ground_truth / torch.exp(x) > self.r
            )  # input is on log scale
            x = x[erasure_indices]
            ground_truth = self.ground_truth[erasure_indices]
            loss = self.kl_loss(
                x.reshape(1, -1), ground_truth.reshape(1, -1)
            )  # erasure loss
        else:
            loss = self.kl_loss(
                x.reshape(1, -1), self.ground_truth.reshape(1, -1)
            )  # kl loss
        return loss
