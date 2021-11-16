#!/bin/bash
for i in {1..32}
do
   python pose_estimation/train_exit.py --cfg  \
   ./experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml --split_point $i
done