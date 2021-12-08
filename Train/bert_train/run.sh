#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./train_output.log"
exec 1>${LOG_FILE} 2>&1

export TASK_NAME=stsb


for i in {1..12}
do
   python run_glue.py \
        --model_name_or_path bert-base-cased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        --split_point $i \
        --output_dir ./models/$TASK_NAME/'exit'$i
done