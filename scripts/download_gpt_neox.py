# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModelForCausalLM, AutoTokenizer

one_million = 1e6
model = AutoModelForCausalLM.from_pretrained(
    f"EleutherAI/gpt-neox-20b", device_map="auto", load_in_8bit=True
)
print(
    f"Done downloading. The model has {model.num_parameters(exclude_embeddings=False)/one_million} million parameters."
)
