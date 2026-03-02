# uses Maltplotlib and worldcloud for visualizations.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


# Saves fig to save_path if provided, shows it if show=True, then closes it.
def _save_or_show(fig, save_path, show):
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# Horizontal bar chart of TF-IDF keyword scores.
def plot_top_keywords(words, scores, *, title="Top Keywords (TF-IDF)", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words, scores)
    ax.set_xlabel("TF-IDF Score")
    ax.set_title(title)
    _save_or_show(fig, save_path, show)


# Plots one bar chart per LDA topic showing the top word probabilities.
def plot_lda_topics(lda_model, *, num_words=8, save_dir=None, show=True):
    for i, topic in lda_model.show_topics(num_words=num_words, formatted=False):
        words = [w for w, _ in topic]
        probs = [p for _, p in topic]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(words, probs)
        ax.set_title(f"Topic {i}")
        ax.set_xticklabels(words, rotation=45, ha="right")
        path = (save_dir / f"topic_{i}.png") if save_dir else None
        _save_or_show(fig, path, show)


# Line plot of coherence score vs. number of LDA topics.
def plot_coherence_curve(topic_nums, scores, *, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(topic_nums, scores, marker="o")
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("Coherence vs Number of Topics")
    _save_or_show(fig, save_path, show)


# Generates and displays/saves a word cloud from a space-joined token string.
def plot_wordcloud(text, *, width=800, height=400, save_path=None, show=True):
    wc = WordCloud(width=width, height=height).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    _save_or_show(fig, save_path, show)
