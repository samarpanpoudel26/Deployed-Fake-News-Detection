import pandas as pd
import nltk
import os
import re
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ======== Download NLTK resources safely & persistently ========
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# ======== Preprocessing Function ========
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    cleaned_text = " ".join(tokens)
    return cleaned_text

# ======== Load + Train Model (without caching for now) ========
def load_model():
    dataset = pd.read_csv("data.xls")
    dataset.drop(columns=["URLs", "Headline"], axis=1, inplace=True)
    dataset.dropna(inplace=True)
    dataset = dataset.sample(frac=1)
    dataset["Body"] = dataset["Body"].apply(preprocessing)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataset["Body"])
    y = dataset["Label"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    model = MultinomialNB()
    model.fit(x_train, y_train)

    return model, vectorizer

# ======== Predict Function Used by app.py ========
def user_input(text):
    model, vectorizer = load_model()
    cleaned = preprocessing(text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)
    return int(prediction[0])
