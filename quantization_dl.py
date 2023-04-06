import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import tqdm
from tqdm.auto import tqdm
import copy
from dataset import data_preparation
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from train_model import simple_train_model

device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=126).to(device)
bert_model.load_state_dict(torch.load('model_bert', map_location=torch.device('cpu')))
bert_model.eval()


def train_one_epoch(model, optimizer, data_loader, device):
    """
    Функция обучения модели на 1 эпоху
    """
    num_training_steps = len(data_loader)
    progress_bar = tqdm(range(num_training_steps))
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)


def move_to_dataloader(df, train_flag=True):
    """
    Функция преобразования в даталоадер
    """
    df = df.rename_column("label", "labels")
    df = df.remove_columns(["summary", 'title'])
    df.set_format("torch")
    dataloader_data = DataLoader(df, shuffle=train_flag, batch_size=16)
    return dataloader_data


def preparation_dataset_quantization(data):
    """
    Функция предобработки данных для обучения
    """
    train_data, eval_data = data_preparation.split_data(data, simple_train_model.tokenize_function)
    train_dataloader = move_to_dataloader(train_data, True)
    eval_dataloader = move_to_dataloader(eval_data, False)
    return train_dataloader, eval_dataloader


def quantization_model(data):
    """
    Функция квантизации модели
    """
    qat_model = copy.deepcopy(bert_model)
    for _, mod in qat_model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

    train_dataloader, eval_dataloader = preparation_dataset_quantization(data)
    qat_model.train()
    optimizer = AdamW(bert_model.parameters(), lr=5e-5)
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    train_one_epoch(qat_model, optimizer, train_dataloader, device=device)
    quantized_model = torch.ao.quantization.convert(qat_model.cpu().eval(), inplace=False)
    torch.save(quantized_model.state_dict(), "bert")
