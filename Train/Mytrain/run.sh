#!/bin/bash
for i in {1..32}
do
   python train_imagenet.py --split_point $i --resume None
done