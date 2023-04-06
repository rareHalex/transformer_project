import re
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def data_load(path):
    """
    Функция загрузки данных с kaggle
    """
    data = pd.read_json(path)
    drop_arr = ['author', 'day', 'id', 'link', 'month', 'year']
    data.drop(columns=drop_arr, axis=1, inplace=True)
    return data


def dict_label_creation(tag_data):
    """
    Функция определения словаря таргетов и их колличества.
    Функция принимает на вход данные о таргете, высчитывает уникальные таргеты,
    создает словари с опредлением таргет : индекс
    """
    unique_label_set = set()
    for labels in tag_data:
        unique_label_set.add(labels)
    count_labels = len(unique_label_set)

    label_dict = {}
    reverse_label_dict = {}
    counter_dict_label = 0
    for caption_title in unique_label_set:
        label_dict[caption_title] = counter_dict_label
        reverse_label_dict[counter_dict_label] = caption_title
        counter_dict_label += 1

    return label_dict, reverse_label_dict, count_labels

def target_preparation(data):
    """
    Функия выделения таргета для предсказания для данных c kaggle.
    Функция принимает датасет и выделяет в нем информацию о названии статьи и аннотацию.
    """
    tag_size = len(data['tag'])
    for captions in range(tag_size):
        arr_tags = data['tag'][captions]
        start_term_indx_arr = [m.start() for m in re.finditer('term', arr_tags)]
        end_term_indx_arr = [m.start() for m in re.finditer('scheme', arr_tags)]
        left_bound = start_term_indx_arr[0]
        right_bound = end_term_indx_arr[0]
        data['tag'][captions] = arr_tags[left_bound:right_bound].split(':')[1][1:-3].replace("'","")

    label_dict, reverse_label_dict, count_labels = dict_label_creation(data['tag'])

    for caption in range(tag_size):
        data['tag'][caption] = label_dict[data['tag'][caption]]
    data.rename(columns={'tag': 'label'}, inplace=True)

    return data, reverse_label_dict, count_labels


def preprocess_text(data, caption):
    """
    Функция обработки текста для обучения
    """
    data[caption] = data[caption].str.lower().replace('[,.;#?!&$]+\ *', ' ', regex=True)
    data[caption] = data[caption].str.replace(r'\s+', ' ', regex=True)
    data[caption] = data[caption].fillna('')
    return data


def split_data(data, tokenize_function):
    """
    Функция разделения и токенизации данных
    """
    train_data, eval_data = train_test_split(data, test_size=0.2)

    train_data = Dataset.from_dict(train_data)
    train_data = train_data.map(tokenize_function, batched=True)

    eval_data = Dataset.from_dict(eval_data)
    eval_data = eval_data.map(tokenize_function, batched=True)
    return train_data, eval_data


def main_preparation():
    """
    основная функция обработки данных с kaggle
    """
    path = ''
    data, label_dict, count_labels = target_preparation(data_load(path))
    data = preprocess_text(data, 'title')
    data = preprocess_text(data, 'summary')
    return data


