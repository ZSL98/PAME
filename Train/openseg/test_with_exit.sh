#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
LOG_FILE="./log/cityscapes/test_output.log"
exec 1>${LOG_FILE} 2>&1

PYTHON="/home/slzhang/miniconda/envs/fedml/bin/python"

# nvidia-smi
# ${PYTHON} -m pip install torchcontrib
# ${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

DATA_ROOT="/home/slzhang/projects/ETBA/Train/openseg/data"
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"
BACKBONE="resnet101_with_only_exit"
# BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="configs/cityscapes/R_101_D_8_with_exit.json"
MODEL_NAME="spatial_ocrnet_with_only_exit"
LOSS_TYPE="fs_auxce_loss"
MAX_ITERS=40000

PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"

# 2>&1 | tee ${LOG_FILE}

for i in {10..30}
do
    CHECKPOINTS_NAME="ocrnet_resnet101_s$i"
    cd /home/slzhang/projects/ETBA/Train/openseg
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_max_performance.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val --data_dir ${DATA_DIR} --split_point $i


    cd /home/slzhang/projects/ETBA/Train/openseg/lib/metrics
    ${PYTHON} -u cityscapes_evaluator.py --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val/label  \
                                       --gt_dir ${DATA_DIR}/val/label 
done