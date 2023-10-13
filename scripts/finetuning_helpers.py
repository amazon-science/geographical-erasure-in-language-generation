# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from tqdm import tqdm


def compute_loss(model, data_loader, loss_function):
    """Computes loss by iterating over all batches and performing forward pass.

    Parameters
    ----------
    model: HF model
    data_loader: data
    loss_function: loss function to be computed

    Returns
    -------
    avg_loss: float, average loss over all batches in `data_loader`
    """
    losses = []
    for prompt in tqdm(data_loader):
        with torch.no_grad():
            log_q = model(prompt, return_logprobs=True)
            loss = loss_function(
                log_q
            )  # no input for target since loss has ground truth already
            losses.append(loss.item())
            avg_loss = np.mean(losses)
    return avg_loss
