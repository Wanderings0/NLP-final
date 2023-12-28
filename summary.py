import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
import evaluate
from transformers import Trainer, TrainingArguments, EvalPrediction
import wandb
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 42
set_seed(seed)

model_name = "t5-base"



wandb.init(project="T5", name="samsum_t5")


# 加载emotion数据集
samsum_dataset = load_from_disk("./data/samsum")

local_model_path = "./model/samsum_tuned_t5"
cond_gen_model_dir = "./model/t5_samsum"
twice = False
if twice:
    print("We fune-tune the model twice")
    cond_gen_model_dir = "./model/emotion_tuned_t5"
    local_model_path = "./model/gen_after_emotion_tuned_t5"

model = T5ForConditionalGeneration(cond_gen_model_dir)
tokenizer = T5Tokenizer.from_pretrained(cond_gen_model_dir, model_max_length=1024, legacy=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def summary_preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

summary_encoded_dataset = samsum_dataset.map(summary_preprocess_function, batched=True)

# 更新训练参数，适用于新的数据集
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=1e-4,
    logging_dir='./logs',
    logging_steps=1,
    report_to="wandb",
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=summary_encoded_dataset["train"],
    eval_dataset=summary_encoded_dataset["validation"],
    tokenizer=tokenizer,
)



trainer.train()
# Make sure the training has finished
trainer.save_model(local_model_path)
tokenizer.save_pretrained(local_model_path)


wandb.join()