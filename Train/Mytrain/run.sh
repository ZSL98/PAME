#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./checkpoints/train_10/train_output_1to10.log"
exec 1>${LOG_FILE} 2>&1

for i in {1..10}
do
   python train_imagenet_only_exit.py --split_point $i --resume None
done