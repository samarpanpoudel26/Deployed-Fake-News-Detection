import streamlit as st
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ======= One-time Downloads (at runtime only when needed) =======
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# ======= Preprocessing Function =======
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

# ======= Training the model (lazy-loaded inside function) =======
@st.cache_resource
def load_model():
    ensure_nltk_data()

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

# ======= User Input Prediction =======
def user_input(text):
    model, vectorizer = load_model()
    ensure_nltk_data()

    cleaned = preprocessing(text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)

    return int(prediction[0])
