#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./log/mpii/train_output.log"
exec 1>${LOG_FILE} 2>&1

for i in {1..33}
do
   python pose_estimation/train.py --cfg  \
   ./experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml --split_point $i
done