from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
    model = AutoModelForSeq2SeqLM.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
    return model, tokenizer


st.title('Page for generating articles and titles')


def title_generation(abstract):
    model, tokenizer = load_model()
    input_ids = tokenizer(abstract, return_tensors="pt").input_ids
    generated = model.generate(input_ids)
    text = tokenizer.decode(generated[0])[5:-5]
    return text


abstract = st.text_area("Write summary of your article")
result_title = st.button('Generation title')
if result_title:
    text = title_generation(abstract)
    st.write(text)

