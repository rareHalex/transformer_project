import torch
import json
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import re

device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=126).to(device)
model.load_state_dict(torch.load('bert', map_location=torch.device('cpu')))
model.eval()

with open('../label.json', 'r') as f:
    target_dict = json.loads(f.read())


def prepare_data_predict(text):
    """
    Предобработка данных для предсказания
    """
    text = text.lower()
    text = re.sub(r"[,.;#?!&$]+\ *", " ", text)
    return text


def prediction_model(title, abstract, strimlit_prediction=False):
    """
    Основная функция предсказания классов статей
    """
    title = prepare_data_predict(title)
    abstract = prepare_data_predict(abstract)
    text = tokenizer(title, abstract, padding="max_length", truncation=True, return_tensors="pt")
    pred = model(**text)
    prediction = torch.softmax(pred['logits'], dim=1)
    prediction *= 100
    predict_sort = prediction.sort(descending=True)
    probs = predict_sort[0][0]
    labels = predict_sort[1][0]

    answer = []
    prob_sum = 0
    idx = 0
    if strimlit_prediction:
        while prob_sum <= 95:
            prob_sum += probs[idx]
            target_of_paper = target_dict[str(int(labels[idx]))]
            if idx <= 5:
                st.write(target_of_paper)
            answer.append(target_of_paper)
            target_dict[target_of_paper] = float(probs[idx])
            idx += 1
            if idx > 6:
                break
        if len(answer) > 2:
            st.write('Probability graph that an article of a given field of science')
            keys = list(target_dict.keys())
            vals = [float(target_dict[k]) for k in keys]
            fig = plt.figure(figsize=(10, 4))
            sns.barplot(x=vals, y=keys)
            st.pyplot(fig)
    else:
        while prob_sum < 95:
            prob_sum += probs[idx]
            answer.append(target_dict[str(int(labels[idx]))])
            idx += 1
            if idx > 6:
                break
    return answer