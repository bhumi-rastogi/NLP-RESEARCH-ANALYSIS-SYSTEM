# LDA topic modeling and coherence evaluation using Gensim.

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


# Builds a Gensim Dictionary and bag-of-words corpus from tokenized documents.
def build_corpus(cleaned_docs):
    dictionary = corpora.Dictionary(cleaned_docs)
    corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]
    return dictionary, corpus


# Trains and returns an LDA model on the given corpus.
def train_lda(corpus, dictionary, num_topics=3, passes=10, random_state=42):
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
    )
    return model


# Computes and returns the coherence score (c_v by default) for a trained LDA model.
def compute_coherence(lda_model, cleaned_docs, dictionary, coherence="c_v"):
    cm = CoherenceModel(
        model=lda_model,
        texts=cleaned_docs,
        dictionary=dictionary,
        coherence=coherence,
    )
    return cm.get_coherence()


# Trains LDA for each topic count in topic_range, returns list of (num_topics, coherence) pairs.
def sweep_num_topics(corpus, dictionary, cleaned_docs, topic_range=range(2, 7), passes=10):
    results = []
    for k in topic_range:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=passes)
        score = compute_coherence(lda, cleaned_docs, dictionary)
        results.append((k, score))
    return results
