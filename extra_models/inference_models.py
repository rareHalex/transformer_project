from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def title_generation(abstract):
    """
    Функция генерации название к тексту
    """
    title_tokenizer = AutoTokenizer.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
    title_model = AutoModelForSeq2SeqLM.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
    input_ids = title_tokenizer(abstract, return_tensors="pt").input_ids
    generated = title_model.generate(input_ids)
    new_title = title_tokenizer.decode(generated[0])[5:-5]
    return new_title


def translation_language(text):
    """
    Функция перевода русского языка на английский
    """
    language_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    language_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    tokenized_text = language_tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    translation = language_model.generate(**tokenized_text)
    translation_text = language_tokenizer.batch_decode(translation, skip_special_tokens=False)[0][5:-4]
    return translation_text


