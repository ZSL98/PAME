#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./train_output.log"
exec 1>${LOG_FILE} 2>&1


for i in {1..12}
do
   python train.py --split_point $i
done