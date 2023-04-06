from transformers import AutoModelForSequenceClassification
from dataset.data_preparation import split_data
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


NUMBER_LABEL = 126
NUMBER_EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=NUMBER_LABEL)


def tokenize_function(examples):
    """
    Функция токенезации для bert
    """
    return tokenizer(examples['summary'], examples['title'], padding="max_length", truncation=True)


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

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(eval_data, batch_size=8)

    return train_dataloader, eval_dataloader


def train_scale_model(data):
    """
    Функция обучения модели
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_data, eval_data = split_data(data, tokenize_function)
    train_dataloader, eval_dataloader=(train_data, eval_data)
    num_training_steps = NUMBER_EPOCHS * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.to(device)
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
    torch.save(model.state_dict(), 'bert')


def to_onnx():
    """
    Функция импорта в стиль ONNX
    """
    s1 = 'Exploring the Solar Poles: The Last Great Frontier of the Sun'
    s2 = "Despite investments in multiple space and ground-based solar observatories."
    text = tokenizer(s1, s2, padding="max_length", truncation=True, return_tensors="pt")
    torch.onnx.export(
        model,
        tuple(text.values()),
        f="model.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                      'attention_mask': {0: 'batch_size', 1: 'sequence'},
                      'logits': {0: 'batch_size', 1: 'sequence'}},
        do_constant_folding=True,
        opset_version=13,
    )
