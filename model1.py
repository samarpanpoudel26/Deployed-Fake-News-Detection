# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# %%
dataset=pd.read_csv("data.xls")

# %%
dataset.head(3)

# %%
dataset.isnull().sum()

# %%
dataset.drop(columns="URLs",axis=1,inplace=True)
dataset.drop(columns="Headline",axis=1,inplace=True)

# %%
dataset.dropna(inplace=True)

# %%
dataset.isnull().sum()

# %%
dataset=dataset.sample(frac=1)

# %%
def preprocessing(text):
    """
    Function that does all the steps of Natural Language Processing.
    
    """
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = " ".join(tokens)
   
    return cleaned_text

# %%
dataset["Body"]=dataset["Body"].apply(preprocessing)

# %%
vectorizer=CountVectorizer()

# %%
X=vectorizer.fit_transform(dataset["Body"])

# %%
Y=dataset[["Label"]]

# %%
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

# %%
mnb=MultinomialNB()
mnb.fit(x_train,y_train)

# %%
mnb.score(x_test,y_test)*100,mnb.score(x_train,y_train)*100

# %%
def user_input(user_text):
    """
    Function to Predict User Input
    """
      
    cleaned_text = preprocessing(user_text)  # Preprocess it
    transformed_text = vectorizer.transform([cleaned_text])  # Convert to numerical format
    prediction = mnb.predict(transformed_text)# Predict
    
    if prediction==1:
        print("Not a fake News")
    else:
        print("Fake news")
    return prediction



