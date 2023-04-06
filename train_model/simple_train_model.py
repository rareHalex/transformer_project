import evaluate
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from dataset.data_preparation import split_data, main_preparation


NUMBER_LABEL = 126
NUMBER_EPOCHS = 5
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tokenize_function(examples):
    """
    Функция токенизации данных
    """
    return tokenizer(examples['summary'], examples['title'], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    """
    Функция подсчета метрики
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model_trainer(data):
    """
    Функция обучения модели с помощью trainer
    """
    train_data, eval_data = split_data(data, tokenize_function)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=NUMBER_LABEL)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    training_args.report_to = ['tensorboard']
    training_args.num_train_epochs = NUMBER_EPOCHS
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics)
    trainer.train()
    torch.save(model.state_dict(), 'distilbert')


if __name__ == "__main__":
    data = main_preparation()
    train_model_trainer(data)
