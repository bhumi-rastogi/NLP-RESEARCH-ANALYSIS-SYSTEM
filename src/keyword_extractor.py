import numpy as np


def extract_keywords(tfidf_matrix, term_names, top_n: int = 15) -> list:
    """
    Rank vocabulary terms by their aggregate TF-IDF weight across
    all documents in the corpus.

    Args:
        tfidf_matrix : sparse matrix from TfidfVectorizer
        term_names   : array of vocabulary strings
        top_n        : how many top-ranked terms to return

    Returns:
        List of (term, score) tuples, sorted descending by score.
    """
    # Sum each term's TF-IDF value across all documents
    aggregated  = np.sum(tfidf_matrix.toarray(), axis=0)
    term_scores = list(zip(term_names, aggregated))
    ranked      = sorted(term_scores, key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]