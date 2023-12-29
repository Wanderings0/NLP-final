import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from datasets import load_dataset, load_from_disk
import evaluate
from transformers import Trainer, TrainingArguments, EvalPrediction, DataCollatorForSeq2Seq
import wandb
import os
import numpy as np
from rouge_score import rouge_scorer, scoring

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
seed = 42
set_seed(seed)

torch.cuda.empty_cache()

model_name = "t5-base"



wandb.init(project="T5", name="samsum_after_emotion_t5")


# 加载emotion数据集
samsum_dataset = load_from_disk("./data/samsum")
# print(samsum_dataset)
# print(samsum_dataset["train"][0])
local_model_path = "./model/samsum_tuned_t5"
cond_gen_model_dir = "./model/t5_cond_gen"
twice = True
if twice:
    print("We fune-tune the model twice")
    cond_gen_model_dir = "./model/emotion_tuned_t5"
    local_model_path = "./model/gen_after_emotion_tuned_t5"

gen_config = T5Config.from_pretrained(cond_gen_model_dir)
model = T5ForConditionalGeneration(gen_config)
pretrained_model = T5ForConditionalGeneration.from_pretrained(cond_gen_model_dir)
model.load_state_dict(pretrained_model.state_dict(),strict=False)

tokenizer = T5Tokenizer.from_pretrained(cond_gen_model_dir, model_max_length=1024, legacy=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=64, truncation=True, padding="max_length")
    model_inputs["length"] = [len(input_ids) for input_ids in model_inputs['input_ids']]
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["label_length"] = [len(input_ids) for input_ids in model_inputs['labels']]
    return model_inputs
summary_encoded_dataset = samsum_dataset.map(preprocess_function, batched=True)
# print an example of summary_data_set
# print(summary_encoded_dataset)
# print(summary_encoded_dataset["train"][0])
# filter by length
# print(len(summary_encoded_dataset["train"].filter(lambda x: x["length"] <= 256)))
# print(len(summary_encoded_dataset["train"].filter(lambda x: x["label_length"] <= 64)))
# summary_encoded_dataset["validation"] = summary_encoded_dataset["validation"].filter(lambda x: x["length"] <= 256)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=3,
    weight_decay=1e-4,
    evaluation_strategy="steps",
    logging_dir='./logs',
    logging_steps=5,
    eval_steps=5,
    report_to="wandb",
    fp16=True,
)


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
aggregate = scoring.BootstrapAggregator()

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    # predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=-1)
    #use 
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print((f"detailed predictions: {decoded_preds[0]}"))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print((f"detailed labels: {decoded_labels[0]}"))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregate = scoring.BootstrapAggregator()

    for ref, pred in zip(decoded_labels, decoded_preds):
        scores = scorer.score(ref, pred)
        aggregate.add_scores(scores)

    final_scores = aggregate.aggregate()
    # 取出中值F1分数输出
    return {
        "rouge1": final_scores["rouge1"].mid.fmeasure,
        "rouge2": final_scores["rouge2"].mid.fmeasure,
        "rougeL": final_scores["rougeL"].mid.fmeasure,
    }

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=summary_encoded_dataset["train"],
    eval_dataset=summary_encoded_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# # 初始化Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=summary_encoded_dataset["train"],
#     eval_dataset=summary_encoded_dataset["validation"],
#     tokenizer=tokenizer,
# )



trainer.train()
# # Make sure the training has finished
trainer.save_model(local_model_path)
tokenizer.save_pretrained(local_model_path)


# wandb.join()