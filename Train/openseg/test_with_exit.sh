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

CHECKPOINTS_NAME="ocrnet_resnet101_s8"
# CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"

${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_max_performance.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val --data_dir ${DATA_DIR}


cd lib/metrics
${PYTHON} -u cityscapes_evaluator.py --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val/label  \
                                       --gt_dir ${DATA_DIR}/val/label