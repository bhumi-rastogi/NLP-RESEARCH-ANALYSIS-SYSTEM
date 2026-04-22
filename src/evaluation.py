from gensim.models import CoherenceModel


def calculate_coherence(lda_model, corpus: list, vocab) -> float:
    """
    Compute the C_v coherence score for a trained LDA model.
    Higher values (typically 0.4–0.7) indicate better topic separation.
    """
    word_lists      = [doc.split() for doc in corpus]
    coherence_eval  = CoherenceModel(
        model      = lda_model,
        texts      = word_lists,
        dictionary = vocab,
        coherence  = "c_v",
    )
    return coherence_eval.get_coherence()