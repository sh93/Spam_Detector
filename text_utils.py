# text_utils.py
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def ensure_nltk_downloads():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    try:
        word_tokenize("Hello world")
    except LookupError:
        nltk.download("punkt")


ensure_nltk_downloads()


def text_cleaner_func(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [
        word
        for word in words
        if word not in stopwords.words("english") and word not in string.punctuation
    ]
    words = [ps.stem(word) for word in words]
    return " ".join(words)
