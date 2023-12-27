import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
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

label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise", 6:"none"}
reverse_label_map = {value: key for key, value in label_map.items()}

wandb.init(project="emotion_finetuning_t5")

# 加载emotion数据集
emotion_dataset = load_dataset("dair-ai/emotion")

# 选择模型和分词器
model_name = "t5-base"
cache_dir = "./cache"
local_model_path = "./emotion_tuned_t5"

if os.path.isdir(local_model_path):
    model = T5ForConditionalGeneration.from_pretrained(local_model_path)
    tokenizer = T5Tokenizer.from_pretrained(local_model_path, model_max_length=1024, legacy=False)
else:
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024, legacy=False, cache_dir=cache_dir)


# lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in emotion_dataset["train"]["text"]]
# max_length = int(np.percentile(lengths, 95))
# print("95% percentile of lengths: {}".format(max_length)) # ouput :51
    
# lengths_labels = [len(tokenizer.encode(label_map[label], add_special_tokens=True)) for label in label_map.keys()]
# print(max(lengths_labels))  # output: 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备emotion数据集，并将文本编码为模型能够处理的格式
def emotion_preprocess_function(examples):
    prefix = "emotion: "
    inputs = [prefix + doc for doc in examples["text"]]
    # 对输入文本进行tokenization
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")

    # 确保labels是字符串列表
    # 假设labels是整数列表，您需要将它们转换为对应的字符串
    # 例如：label_map = {0: "sadness", 1: "joy", 2: "anger", ...}
    str_labels = [label_map[label] for label in examples["label"]]

    labels = tokenizer(str_labels, max_length=4, truncation=True, padding="max_length")

    # 将处理后的标签添加到model_inputs中
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
emotion_encoded_dataset = emotion_dataset.map(emotion_preprocess_function, batched=True, remove_columns=["text", "label"])
# print a sample from the encoded dataset
# print(emotion_encoded_dataset["train"][0])
# print(emotion_encoded_dataset)

input_ids = emotion_encoded_dataset["train"]["input_ids"]
if isinstance(input_ids, list) and len(input_ids) > 0 and not isinstance(input_ids[0], list):
    print("Converting input_ids to a list of lists")
    # 如果input_ids是一维列表，将其转换为二维列表
    emotion_encoded_dataset["train"]["input_ids"] = [input_ids]

def compute_metrics(p: EvalPrediction):
    # 转换预测结果的token ids到字符串
    pred_str = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
    # 转换真实标签的token ids到字符串
    label_str = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)

    # 反转label_map字典，从字符串标签映射到整数

    # 初始化预测和真实标签的整数列表
    pred_labels = [reverse_label_map[label] if label in reverse_label_map.keys() else 6 for label in pred_str]
    true_labels = [reverse_label_map[label] for label in label_str]

    # 计算accuracy 和 micro F1 score
    accuracy = accuracy_score(true_labels, pred_labels)
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    
    return {
        'accuracy': accuracy,
        'micro_f1': micro_f1,
    }

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
    logging_steps=15,
    report_to="wandb",
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
trainer.evaluate()

# Evaluate on the test dataset
def evaluate_test_dataset(trainer, test_dataset):
    # Predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    # Decode predictions
    pred_str = tokenizer.batch_decode(predictions.argmax(-1), skip_special_tokens=True)
    # Decode true labels
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Convert string labels to integers
    pred_labels = [reverse_label_map[label] if label in reverse_label_map else 6 for label in pred_str]
    true_labels = [reverse_label_map[label] for label in label_str]
    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    return accuracy, micro_f1

# Evaluate the model on the test dataset
test_accuracy, test_micro_f1 = evaluate_test_dataset(trainer, emotion_encoded_dataset["test"])

# Log the metrics to wandb
wandb.log({"test_accuracy": test_accuracy, "test_micro_f1": test_micro_f1})

# 保存模型
trainer.save_model("./emotion_tuned_t5")

wandb.join()