# NLP-final Project: Can We Teach a Model Twice

## Preparation

### Packages

Install the packages in the requirements.txt file with:

```
pip install -r requirements.txt
```

### Load Data and Models

Load Data and Models from Huggingface. You can modify the path in the code to change the location of the data and models.

```
python load_data.py
python load_model.py
```
The dataset we used are listed below:

| Dataset         | Train Size | Type               |
|-----------------|------------|--------------------|
| dair-ai/emotion | 20k rows   | Text Classification|
| ag-news         | 120k rows  | Text Classification|
| samsum          | 14.7k rows | Summarization      |

The model we used as backbone is T5-base. 

## Run the Code

### Train the Model on the dair-ai/emotion dataset first, twice is not allowed.

```
python emotion.py --model_name = t5-base --twice = False --lr = 1e-4 --batch_size = 32 --seed = 42 --epochs = 3
```

The arguments are listed below:

| Argument     | Type  | Default  | Description                               |
|--------------|-------|----------|-------------------------------------------|
| --model_name | str   | t5-base  | the model name of the seq_class model     |
| --twice      | bool  | False    | whether to fine-tune the model twice      |
| --lr         | float | 1e-4     | the learning rate of the model            |
| --batch_size | int   | 32       | the batch size of the model               |
| --seed       | int   | 42       | the seed of the model                     |
| --epoch      | int   | 3        | the epoch of the model                    |

### Train the Model on the ag-news dataset, twice is allowed.

``` 
python news.py --model_name = t5-base --twice = True --lr = 1e-4 --batch_size = 32 --seed = 42 --epochs = 1
```

### Train the Model on the samsum dataset, twice is allowed.

```
python samsum.py --model_name = t5-base --twice = True --lr = 2e-5 --batch_size = 24 --seed = 42 --epochs = 3
```


