# #!/bin/bash
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# LOG_FILE="./log/cityscapes/train_output_finetune.log"
# exec 1>${LOG_FILE} 2>&1

PYTHON="/home/slzhang/miniconda/envs/fedml/bin/python"

# nvidia-smi
# ${PYTHON} -m pip install torchcontrib
# ${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

DATA_ROOT="/home/slzhang/projects/ETBA/Train/openseg/data"
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"
CONFIGS="configs/cityscapes/R_101_D_8_with_exit.json"
CHECKPOINTS_NAME="ocrnet_resnet101_s"
# CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"

MAX_ITERS=40000
# BACKBONE="resnet101_with_only_exit"
BACKBONE="resnet101_with_only_exit"
MODEL_NAME="spatial_ocrnet_with_multi_exit"
LOSS_TYPE="fs_auxce_loss_with_multi_exit"


# TODO: WARNING!!!!!!!! change checkpoints_dir in the json file


python -u main.py --configs ${CONFIGS} \
                    --drop_last y \
                    --phase etrain \
                    --gathered n \
                    --loss_balance y \
                    --log_to_file n \
                    --backbone ${BACKBONE} \
                    --model_name ${MODEL_NAME} \
                    --gpu 0 \
                    --data_dir ${DATA_DIR} \
                    --loss_type ${LOSS_TYPE} \
                    --max_iters ${MAX_ITERS} \
                    --checkpoints_name ${CHECKPOINTS_NAME} \
                    --pretrained ${PRETRAINED_MODEL} \
                    --distributed \
                    --split_point 888 \
                    2>&1 | tee ${LOG_FILE}

# for i in 10 13
# do
#     # TODO: log results
#     python -u main.py --configs ${CONFIGS} \
#                        --drop_last y \
#                        --phase etrain \
#                        --gathered n \
#                        --loss_balance y \
#                        --log_to_file n \
#                        --backbone ${BACKBONE} \
#                        --model_name ${MODEL_NAME} \
#                        --gpu 0 \
#                        --data_dir ${DATA_DIR} \
#                        --loss_type ${LOSS_TYPE} \
#                        --max_iters ${MAX_ITERS} \
#                        --checkpoints_name ${CHECKPOINTS_NAME} \
#                        --pretrained ${PRETRAINED_MODEL} \
#                        --distributed \
#                        --split_point $i \
#                        2>&1 | tee ${LOG_FILE}
# done