#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./checkpoints/train_metric_controlled/train_output.log"
exec 1>${LOG_FILE} 2>&1

for i in 7 10 13 16 19 22 25 28 31
do
   python train_imagenet_only_exit.py --split_point $i --resume None
done