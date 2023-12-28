import torch
from transformers import T5ForSequenceClassification, T5Tokenizer
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


label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
reverse_label_map = {value: key for key, value in label_map.items()}
num_labels = len(label_map)

wandb.init(project="T5", name="emotion_after_news_tuned_t5")


# 加载emotion数据集
emotion_dataset = load_from_disk("./data/emotion")

local_model_path = "./model/emotion_tuned_t5"
seq_class_model_dir = "./model/t5_seq_class"
twice = True
if twice:
    print("We fune-tune the model twice")
    seq_class_model_dir = "./model/news_tuned_t5"
    local_model_path = "./model/emotion_after_news_tuned_t5"
model = T5ForSequenceClassification.from_pretrained(seq_class_model_dir,num_labels=num_labels,ignore_mismatched_sizes=True)
tokenizer = T5Tokenizer.from_pretrained(seq_class_model_dir, model_max_length=1024, legacy=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def emotion_preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=64, truncation=True,padding="max_length")
    model_inputs["labels"] = examples["label"]
    return model_inputs
emotion_encoded_dataset = emotion_dataset.map(emotion_preprocess_function, batched=True)

# sample a example from emotion_encoded_dataset
# print(emotion_encoded_dataset["train"][0])

# def the compute metrics function
def compute_metrics(p: EvalPrediction):
    # logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    logits = p.predictions[0]
    softmax_logits = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    
    # Convert logits to numpy and take argmax to get predicted class indices
    preds = np.argmax(softmax_logits.numpy(), axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=4,
    evaluation_strategy="steps",
    logging_strategy="steps",
    eval_steps=4,
    report_to="wandb",
    seed=42,
    load_best_model_at_end=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotion_encoded_dataset["train"],
    eval_dataset=emotion_encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始emotion数据集的fine-tuning
trainer.train()
# Make sure the training has finished
trainer.save_model(local_model_path)
tokenizer.save_pretrained(local_model_path)


wandb.join()