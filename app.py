import streamlit as st
import pickle
import pandas as pd

import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

ps = PorterStemmer()


def transform_data(text, **args):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") or i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def text_cleaner_func(X):
    return X.apply(transform_data)


model_path = "RFCModel.pkl"

st.title("Spam Detector")
message_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if len(message_input.strip()) <= 1:
        st.error("Enter a valid message.")
    else:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        prediction = model.predict(pd.Series([message_input]))[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        if prediction == 0:
            st.success("Not Spam")
        else:
            st.warning("Spam")
