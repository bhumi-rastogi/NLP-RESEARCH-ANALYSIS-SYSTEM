import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK assets if not already present
_required_packages = {
    "punkt":     "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}
for _pkg, _path in _required_packages.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# Load spaCy model and English stopword list once at module level
_nlp_model   = spacy.load("en_core_web_sm")
_stop_words  = set(stopwords.words("english"))


def _normalize_text(raw: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    lowered  = raw.lower()
    no_punct = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", no_punct).strip()


def _split_tokens(text: str) -> list:
    """Tokenize text into individual word tokens."""
    return word_tokenize(text)


def _drop_stopwords(token_list: list) -> list:
    """Remove stopwords and non-alphabetic tokens."""
    return [t for t in token_list if t.isalpha() and t not in _stop_words]


def _apply_lemmatization(token_list: list) -> list:
    """Reduce each token to its dictionary base form via spaCy."""
    joined = " ".join(token_list)
    parsed = _nlp_model(joined)
    return [tok.lemma_ for tok in parsed if tok.lemma_.isalpha()]


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    normalize → tokenize → remove stopwords → lemmatize.
    Returns a single string of processed tokens.
    """
    normalized = _normalize_text(text)
    tokens     = _split_tokens(normalized)
    tokens     = _drop_stopwords(tokens)
    tokens     = _apply_lemmatization(tokens)
    return " ".join(tokens)


def get_text_stats(text: str) -> dict:
    """Compute surface-level statistics over raw input text."""
    all_sentences  = sent_tokenize(text)
    all_words      = word_tokenize(text)
    alpha_words    = [w for w in all_words if w.isalpha()]
    unique_vocab   = set(w.lower() for w in alpha_words)
    return {
        "sentences":    len(all_sentences),
        "words":        len(alpha_words),
        "unique_tokens": len(unique_vocab),
    }