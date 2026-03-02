# Extractive summarization: ranks sentences by TF-IDF score and returns the top-n.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_research.preprocessing.text_preprocessor import split_into_sentences


# Returns the top-n most informative sentences from text, in original document order.
def summarize(text, top_n=5):
    sentences = split_into_sentences(text)
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)
    scores = np.array(np.sum(sentence_vectors, axis=1)).flatten()
    top_indices = sorted(np.argsort(scores)[-top_n:].tolist())
    return [sentences[i] for i in top_indices]
