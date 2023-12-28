from transformers import T5ForConditionalGeneration, T5Tokenizer, T5ForSequenceClassification,Trainer
import os

# 确保model目录存在
model_dir = "./model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 指定模型名称
model_name = "t5-base"

# 下载T5ForConditionalGeneration模型和tokenizer
cond_gen_model = T5ForConditionalGeneration.from_pretrained(model_name)
cond_gen_tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=1024, truncation=True,legacy=False)
# cond_gen_tokenizer = T5Tokenizer.from_pretrained('t5-base',model_max_length=1024, truncation=True,legacy=False)
cond_gen_model.save_pretrained(os.path.join(model_dir, "t5_cond_gen"))
cond_gen_tokenizer.save_pretrained(os.path.join(model_dir, "t5_cond_gen"))
# # 加载 T5 模型
seq_class_model = T5ForSequenceClassification.from_pretrained('t5-base')
seq_class_tokenizer = T5Tokenizer.from_pretrained('t5-base',model_max_length=1024, truncation=True,legacy=False)
# 保存T5ForConditionalGeneration模型和tokenizer
seq_class_model.save_pretrained(os.path.join(model_dir, "t5_seq_class"))
seq_class_tokenizer.save_pretrained(os.path.join(model_dir, "t5_seq_class"))
print("模型和tokenizer下载并保存成功。")