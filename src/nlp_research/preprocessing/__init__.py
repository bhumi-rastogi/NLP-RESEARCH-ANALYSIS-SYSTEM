"""Preprocessing sub-package: tokenization, stopword removal, lemmatization."""
from .text_preprocessor import preprocess, split_into_sentences, split_into_paragraphs

__all__ = ["preprocess", "split_into_sentences", "split_into_paragraphs"]
