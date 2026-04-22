# LLM-powered summarization using Google Gemini.
# Falls back to TextRank extractive summarization if no API key is provided.

import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_research.preprocessing.text_preprocessor import split_into_sentences


# ──────────────────────────────────────────────────────────────
# Extractive fallback (TextRank-inspired)
# ──────────────────────────────────────────────────────────────

def _is_good_sentence(s, min_words=6, max_words=70):
    s = s.strip()
    words = s.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if re.search(r"(https?://|www\.|\[\d+\])", s):
        return False
    non_alpha = sum(1 for c in s if not c.isalpha() and not c.isspace())
    if non_alpha > len(s) * 0.4:
        return False
    return True


def _textrank_scores(sentences):
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vecs = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.ones(len(sentences))
    sim = cosine_similarity(vecs)
    np.fill_diagonal(sim, 0)
    return sim.sum(axis=1)


def _extractive_summarize(text, top_n=5):
    """TextRank-based extractive summarization (no API required)."""
    sentences = split_into_sentences(text)
    good = [(i, s.strip()) for i, s in enumerate(sentences) if _is_good_sentence(s)]
    if not good:
        return [s.strip() for s in sentences[:top_n] if s.strip()]
    if len(good) <= top_n:
        return [s for _, s in good]
    indices, good_sentences = zip(*good)
    scores = _textrank_scores(list(good_sentences))
    top_local = sorted(np.argsort(scores)[-top_n:].tolist())
    return [good_sentences[i] for i in top_local]


# ──────────────────────────────────────────────────────────────
# LLM summarization via Google Gemini
# ──────────────────────────────────────────────────────────────

def _gemini_summarize(text, top_n=5, api_key=None):
    """
    Use Google Gemini to generate a clean abstractive summary.
    Returns a list of bullet-point sentences.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai is not installed. Run: pip install google-generativeai")

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("No Gemini API key provided.")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Truncate text to avoid token limits (~30k chars ≈ ~7500 tokens)
    truncated = text[:30000]

    prompt = f"""You are an expert research assistant. Read the following document and write a concise, \
clear summary in exactly {top_n} bullet points. Each bullet point must be a complete, standalone sentence \
that captures a key finding or contribution. Do NOT use vague phrases like "the paper discusses". \
Be specific and informative.

Document:
{truncated}

Summary ({top_n} bullet points):"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Parse bullet points into a list of sentences
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    bullets = []
    for line in lines:
        # Strip leading bullet markers: •, -, *, 1., 2. etc.
        cleaned = re.sub(r"^[\*\-•\d]+[\.\)]\s*", "", line).strip()
        if cleaned:
            bullets.append(cleaned)

    return bullets[:top_n] if bullets else [raw]


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def summarize(text, top_n=5, api_key=None, use_llm=False):
    """
    Summarize text.
    - If use_llm=True and api_key is provided, uses Google Gemini (abstractive).
    - Otherwise falls back to TextRank extractive summarization.
    Returns a list of sentence strings.
    """
    if use_llm and api_key:
        try:
            return _gemini_summarize(text, top_n=top_n, api_key=api_key)
        except Exception as e:
            # Gracefully fall back to extractive on any error
            return _extractive_summarize(text, top_n=top_n) + [f"⚠️ LLM error: {e}"]
    return _extractive_summarize(text, top_n=top_n)
