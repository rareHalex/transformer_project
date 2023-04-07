import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import re
import json
import predict_model
device = 'cpu'


@st.cache(allow_output_mutation=True)
def language_model():
    language_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    language_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    return language_model, language_tokenizer


@st.cache(allow_output_mutation=True)
def load_data():
    with open('./label.json', 'r') as f:
        target_dict = json.loads(f.read())

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=126).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model.load_state_dict(torch.load('bert', map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer, target_dict


def prepare_data(text):
    text = text.lower()
    text = re.sub(r"[,.;#?!&$]+\ *", " ", text)
    return text


def preapare_language(text, language_tokenizer, language_model):
    tokenized_text = language_tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    translation = language_model.generate(**tokenized_text)
    text = language_tokenizer.batch_decode(translation, skip_special_tokens=False)[0][5:-4]
    return text


def inference_sweets():
    HtmlFile = open("index.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code)




inference_sweets()
st.write('Select a language/ default is English')
rus_buton = st.button('Russian')

if rus_buton:

    title = st.text_area("Write the title of the article")
    summary = st.text_area("Write an abstract of the article")
    language_model, language_tokenizer = language_model()

    title = preapare_language(title, language_tokenizer, language_model)
    summary = preapare_language(summary, language_tokenizer, language_model)

    result_button = st.button('Click')
    if result_button:
        st.write('The article is written in the field of science:')
        _ = predict_model.predict(title, summary, True)
else:

    title = st.text_area("Write the title of the article")
    summary = st.text_area("Write an abstract of the article")

    result_button = st.button('Click')
    if result_button:
        st.write('The article is written in the field of science:')
        _ = predict_model.predict(title, summary, True)
