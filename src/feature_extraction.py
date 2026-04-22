from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(documents: list):
    """
    Fit a TF-IDF vectorizer on the provided document corpus.
    Returns the sparse matrix, array of feature names, and the fitted vectorizer.
    """
    vec    = TfidfVectorizer(max_df=1.0, min_df=1)
    matrix = vec.fit_transform(documents)
    terms  = vec.get_feature_names_out()
    return matrix, terms, vec