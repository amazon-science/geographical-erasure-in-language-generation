#!/bin/sh
MODEL_TYPES=('gpt2' 'gpt2-medium' 'gpt2-large'  'gpt2-xl')
MODEL_TYPES_GPT_NEO=("gpt-neo-125M" "gpt-neo-1.3B" "gpt-neo-2.7B")
MODEL_TYPES_NEOX=("gpt-neox-20b")

for MODEL in {0..0}
do
    python compute_predictions.py --model_type ${MODEL_TYPES[$MODEL]} --gpu $MODEL --experiment_folder 'size_exp'
done