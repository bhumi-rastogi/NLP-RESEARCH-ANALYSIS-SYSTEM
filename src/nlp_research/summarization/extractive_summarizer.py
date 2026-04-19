# Extractive summarization using a TextRank-inspired sentence similarity approach.
# Sentences that share key vocabulary with many other sentences score higher,
# producing coherent summaries rather than isolated jargon-heavy fragments.

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_research.preprocessing.text_preprocessor import split_into_sentences


def _is_good_sentence(s, min_words=6, max_words=70):
    """Filter sentences that are too short, too long, or noisy."""
    s = s.strip()
    words = s.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    # Skip lines that look like headers, URLs, or citation markers
    if re.search(r"(https?://|www\.|\[\d+\])", s):
        return False
    # Skip lines that are mostly numbers (table rows, page numbers, etc.)
    non_alpha = sum(1 for c in s if not c.isalpha() and not c.isspace())
    if non_alpha > len(s) * 0.4:
        return False
    return True


def _textrank_scores(sentences):
    """
    Build a sentence similarity matrix using TF-IDF cosine similarity,
    then score each sentence by the sum of its similarities to all others.
    This mirrors the PageRank idea: sentences 'voted for' by many similar
    sentences are more central to the document's meaning.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vecs = vectorizer.fit_transform(sentences)
    except ValueError:
        # Fallback: uniform scores
        return np.ones(len(sentences))

    sim_matrix = cosine_similarity(vecs)
    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, 0)
    # Score = sum of similarities to all other sentences
    scores = sim_matrix.sum(axis=1)
    return scores


def summarize(text, top_n=5):
    """Return the top-n most representative sentences in document order."""
    sentences = split_into_sentences(text)

    # Filter to readable sentences only
    good = [(i, s.strip()) for i, s in enumerate(sentences) if _is_good_sentence(s)]

    if not good:
        # Fallback: return first top_n raw sentences (cleaned)
        return [s.strip() for s in sentences[:top_n] if s.strip()]

    if len(good) <= top_n:
        return [s for _, s in good]

    indices, good_sentences = zip(*good)
    good_sentences = list(good_sentences)

    scores = _textrank_scores(good_sentences)

    # Pick top_n by score, then restore original document order
    top_local = np.argsort(scores)[-top_n:]
    top_local_sorted = sorted(top_local.tolist())

    return [good_sentences[i] for i in top_local_sorted]
