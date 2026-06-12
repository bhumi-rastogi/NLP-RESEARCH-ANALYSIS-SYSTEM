import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def summarize_text(text: str, num_sentences: int = 5):
    """
    Extractive text summarization using TF-IDF sentence scoring.

    Each sentence is scored by summing its TF-IDF term weights.
    The top-scoring sentences are selected and returned in their
    original document order to preserve readability.

    Args:
        text          : raw input text to summarize
        num_sentences : maximum number of sentences to include

    Returns:
        Tuple of (summary_string, list_of_(sentence, score)_pairs)
    """
    sentence_list = nltk.sent_tokenize(text)

    if not sentence_list:
        return "", []

    # Clamp to the actual number of available sentences
    target_count = min(num_sentences, len(sentence_list))

    if len(sentence_list) == 1:
        return sentence_list[0], [(sentence_list[0], 1.0)]

    try:
        vectorizer    = TfidfVectorizer(stop_words="english")
        sent_matrix   = vectorizer.fit_transform(sentence_list)
        sent_scores   = np.sum(sent_matrix.toarray(), axis=1)
    except ValueError:
        # Edge case: all sentences contain only stop words
        fallback = " ".join(sentence_list[:target_count])
        return fallback, []

    # Pick highest-scoring indices, then sort to restore document order
    top_indices     = np.argsort(sent_scores)[::-1][:target_count]
    ordered_indices = sorted(top_indices)

    chosen_sentences = [sentence_list[i] for i in ordered_indices]
    sentence_scores  = [(sentence_list[i], float(sent_scores[i])) for i in ordered_indices]

    return " ".join(chosen_sentences), sentence_scores