from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric

raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

# trainer = Trainer(
#     model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
# )

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.evaluate()

trainer.train()