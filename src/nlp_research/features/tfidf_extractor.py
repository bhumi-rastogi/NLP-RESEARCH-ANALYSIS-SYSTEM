# TF-IDF feature extraction.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Fits TF-IDF on a list of pre-processed text strings, returns (matrix, vectorizer).
def build_tfidf_matrix(processed_texts, max_df=1.0, min_df=1):
    # Filter out empty/non-string entries
    processed_texts = [t for t in processed_texts if isinstance(t, str) and t.strip()]
    if not processed_texts:
        raise ValueError("No valid text documents after preprocessing — cannot build TF-IDF matrix.")
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    try:
        matrix = vectorizer.fit_transform(processed_texts)
    except ValueError as e:
        raise ValueError(
            f"TF-IDF vocabulary is empty. The document may be too short or contain only stop words. Details: {e}"
        ) from e
    return matrix, vectorizer


# Returns top-N (word, score) pairs for a given document row, sorted by score descending.
def get_top_keywords(tfidf_matrix, vectorizer, doc_index=0, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix[doc_index].toarray().flatten()
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(feature_names[i], float(scores[i])) for i in top_indices]
