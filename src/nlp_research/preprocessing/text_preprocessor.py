# Text cleaning pipeline: lowercase -> tokenize -> stopword removal -> lemmatize.

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# Cleans and lemmatizes raw text, returns a list of tokens.
def preprocess(text, min_word_len=3):
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    cleaned = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha()
        and word not in _stop_words
        and len(word) >= min_word_len
    ]
    return cleaned


# Splits raw text into individual sentences using NLTK sent_tokenize.
def split_into_sentences(text):
    if not text:
        return []
    return sent_tokenize(text)


# Splits text on newlines and discards paragraphs shorter than min_length chars.
def split_into_paragraphs(text, min_length=50):
    if not text:
        return []
    paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > min_length]
