import gensim
from gensim import corpora


def build_lda_model(text_corpus: list, num_topics: int = 5):
    """
    Train an LDA topic model on the supplied text corpus.

    Steps:
      1. Tokenize each document by whitespace.
      2. Build a gensim Dictionary mapping words to IDs.
      3. Convert each document to bag-of-words format.
      4. Train LDA with the specified number of topics.

    Returns:
      (lda_model, topic_list, bow_corpus, dictionary)
    """
    tokenized_docs = [doc.split() for doc in text_corpus]
    vocab          = corpora.Dictionary(tokenized_docs)
    bow_corpus     = [vocab.doc2bow(doc) for doc in tokenized_docs]

    lda = gensim.models.LdaModel(
        corpus     = bow_corpus,
        id2word    = vocab,
        num_topics = num_topics,
        passes     = 10,
    )

    topic_list = lda.print_topics()
    return lda, topic_list, bow_corpus, vocab