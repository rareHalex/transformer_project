import tensor_parallel as tp
from transformers import BloomForSequenceClassification,BloomTokenizerFast
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from dataset.data_preparation import split_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m", add_prefix_space=True)
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", num_labels=126, problem_type="multi_label_classification", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = tp.tensor_parallel(model)  # преобразуем для обучения
NUMBER_EPOCHS = 1


def tokenize_function(examples):
    """
    Функция токенезации данных для BLOOM
    """
    return tokenizer(examples['summary'], examples['title'], padding="max_length", truncation=True, max_length=3800)


def data_prepare(train_data, eval_data):
    """
    Функция подготовки данных к обучению
    """
    train_data = train_data.rename_column("label", "labels")
    eval_data = eval_data.rename_column("label", "labels")

    train_data = train_data.remove_columns(["summary", 'title'])
    eval_data = eval_data.remove_columns(["summary", 'title'])

    train_data.set_format("torch")
    eval_data.set_format("torch")

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=2)  # только с таким размером батча влезает
    eval_dataloader = DataLoader(eval_data, batch_size=2)  # только с таким размером батча влезает

    return train_dataloader, eval_dataloader


def train_large_model(data):
    """
    Функция обучения модели
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_data, eval_data = split_data(data, tokenize_function)
    train_dataloader, eval_dataloader = data_prepare(train_data, eval_data)
    num_training_steps = NUMBER_EPOCHS * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(NUMBER_EPOCHS):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.update(1)
    torch.save(model.state_dict(), 'bloom')
