import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import wandb

summary_dataset = load_dataset("cnn_dailymail", "3.0.0")
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=1024,legacy=False)
# 准备cnn_dailymail数据集，并将文本编码为模型能够处理的格式
def summary_preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

summary_encoded_dataset = summary_dataset.map(summary_preprocess_function, batched=True)

# 更新训练参数，适用于新的数据集
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
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

# 开始cnn_dailymail数据集的fine-tuning
trainer.train()

# 保存模型
trainer.save_model("./summary_tuned_t5")
