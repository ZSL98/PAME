# choose from 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'
TASK_NAME='mrpc'
CKPT_DIR=tmp/model/glue/mrpc/debug/seed_42_12-21-15-36/

SEED=42

EXP_PREFIX=debug
OUTPUT_DIR=tmp/model/glue/${TASK_NAME}/${EXP_PREFIX}/eval_output_seed_${SEED}

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

mkdir -p ${OUTPUT_DIR}

python -u src/run_glue_no_trainer.py \
  --model_type etba \
  --augment_layer 5\
  --model_name_or_path ${CKPT_DIR} \
  --tokenizer_name bert-base-uncased \
  --do_evaluate \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_eval_batch_size 32 \
  --seed ${SEED} \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log.log