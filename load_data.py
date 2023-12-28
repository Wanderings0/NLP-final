from datasets import load_dataset

# 数据集名称和路径
datasets_to_download = {
    'samsum': 'samsum'
}

# 保存目录
save_directory = './data'

for dataset_name, dataset_path in datasets_to_download.items():
    # 加载数据集
    dataset = load_dataset(dataset_path)

    # 保存数据集到指定目录
    dataset.save_to_disk(f'{save_directory}/{dataset_name}')

print("Datasets are downloaded and saved.")