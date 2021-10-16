from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutput
# from wav2vec2_model import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config, Wav2Vec2Encoder, Wav2Vec2EncoderLayer
from wav2vec2_model import Wav2Vec2_with_exit
import torch
import torch.nn as nn
import copy
from prepare_dataset import data_collator, compute_metrics, processor, prepare_dataset
import os
import random
import numpy as np

if __name__ == '__main__':

    dataset_preparation = prepare_dataset()
    dataset_preparation.data_preprocess()
    timit_prepared = dataset_preparation.prepared_dataset

    model_CTC = Wav2Vec2_with_exit.from_pretrained("facebook/wav2vec2-base-960h")
    model_CTC.add_exit(5)
    model_CTC.freeze_feature_extractor()

    training_args = TrainingArguments(
    # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
    output_dir="./wav2vec2-base-timit-demo",
    group_by_length=True,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    )

    trainer = Trainer(
        model=model_CTC,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit_prepared["train"],
        eval_dataset=timit_prepared["test"],
        tokenizer=processor.feature_extractor,
    )

    os.environ["WANDB_DISABLED"] = "true"
    trainer.train()