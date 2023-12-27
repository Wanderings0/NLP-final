import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import wandb

# 初始化Weights & Biases
wandb.init(project="emotion_finetuning_t5")

# 加载emotion数据集
emotion_dataset = load_dataset("dair-ai/emotion")

# 选择模型和分词器
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=1024,legacy=False)

# 准备emotion数据集，并将文本编码为模型能够处理的格式
def emotion_preprocess_function(examples):
    prefix = "emotion: "
    inputs = [prefix + doc for doc in examples["text"]]
    # 对输入文本进行tokenization
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # 确保labels是字符串列表
    # 假设labels是整数列表，您需要将它们转换为对应的字符串
    # 如果您有一个映射表将整数标签映射到对应的字符串标签，您需要在这里使用它
    # 例如：label_map = {0: "sadness", 1: "joy", 2: "anger", ...}
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    str_labels = [label_map[label] for label in examples["label"]]
    # str_labels = [str(label) for label in examples["label"]]
    # print(str_labels)
    # print(type(str_labels),type(str_labels[0]))
    # 对字符串类型的标签进行tokenization，使用text_target参数
    labels = tokenizer(str_labels, max_length=16, truncation=True, padding="max_length")

    # 将处理后的标签添加到model_inputs中
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
emotion_encoded_dataset = emotion_dataset.map(emotion_preprocess_function, batched=True, remove_columns=["text", "label"])
# 确保input_ids等关键输入有两个维度
input_ids = emotion_encoded_dataset["train"]["input_ids"]
if isinstance(input_ids, list) and len(input_ids) > 0 and not isinstance(input_ids[0], list):
    print("Converting input_ids to a list of lists")
    # 如果input_ids是一维列表，将其转换为二维列表
    emotion_encoded_dataset["train"]["input_ids"] = [input_ids]
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
    train_dataset=emotion_encoded_dataset["train"],
    eval_dataset=emotion_encoded_dataset["validation"],
    tokenizer=tokenizer,
)

# 开始emotion数据集的fine-tuning
trainer.train()
test_dataset = load_dataset("dair-ai/emotion", split="test")
test_encoded_dataset = test_dataset.map(emotion_preprocess_function, batched=True, remove_columns=["text", "label"])
trainer.evaluate(test_encoded_dataset["test"])

# 保存模型
trainer.save_model("./emotion_tuned_t5")

# 加载cnn_dailymail数据集
summary_dataset = load_dataset("cnn_dailymail", "3.0.0")

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
training_args.output_dir = "./summary_results"
training_args.num_train_epochs = 5  # 可能需要更多的epoch进行summary任务

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

# 关闭W&B的run
wandb.finish()