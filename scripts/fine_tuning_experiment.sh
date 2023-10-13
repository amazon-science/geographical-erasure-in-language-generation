#!/bin/sh
MODEL_TYPES=('gpt2') # 'gpt2-medium' 'gpt2-large'  'gpt2-xl')
GPUS=(1 2 3)
SPLIT_STRATEGIES=('random' 'pronouns' 'verbs')
r=1

for ix in {0..2}
do
    for fold in {4..4}
    do
        python finetune.py --model_type 'gpt2' --gpu ${GPUS[$ix]} \
        --learning_rate 3e-5 --train_test_split_strategy  ${SPLIT_STRATEGIES[$ix]} --experiment_folder r="$r" \
        --use_erasure True --fold $fold --r $r &
    done
done