#!/bin/sh
MODEL_TYPES=('open_llama_7b' 'open_llama_3b')

for MODEL in {0..0}
do
    python compute_predictions.py --model_type ${MODEL_TYPES[$MODEL]} 
done