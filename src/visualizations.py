import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Apply a consistent dark theme across all matplotlib figures
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
})


def generate_wordcloud(processed_text: str):
    """
    Render a word cloud from pre-processed text.
    Returns a matplotlib Figure object ready for st.pyplot().
    """
    cloud = WordCloud(
        width              = 900,
        height             = 420,
        background_color   = "#0d1117",
        colormap           = "plasma",
        max_words          = 100,
        prefer_horizontal  = 0.85,
        collocations       = False,
    ).generate(processed_text)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Frequency Cloud", fontsize=14, pad=10, color="#c9d1d9")
    fig.tight_layout()
    return fig


def plot_top_keywords(keyword_pairs: list):
    """
    Draw a horizontal bar chart ranking keywords by TF-IDF weight.
    Accepts a list of (word, score) tuples.
    """
    if not keyword_pairs:
        return None

    labels, values = zip(*keyword_pairs)
    palette        = plt.cm.plasma(np.linspace(0.25, 0.88, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=palette)
    ax.invert_yaxis()
    ax.set_xlabel("TF-IDF Weight")
    ax.set_title("Keyword Relevance Ranking", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_topic_distribution(topic_list: list):
    """
    Create a side-by-side bar chart showing the top words for each LDA topic.
    topic_list: list of (topic_id, topic_string) returned by lda_model.print_topics().
    """
    if not topic_list:
        return None

    n_topics   = len(topic_list)
    color_pool = plt.cm.tab10.colors

    fig, axes  = plt.subplots(1, n_topics, figsize=(4 * n_topics, 4), sharey=False)
    if n_topics == 1:
        axes = [axes]

    for ax, (t_id, t_str) in zip(axes, topic_list):
        word_weight_pairs = []
        for segment in t_str.split(" + "):
            try:
                weight, word = segment.split('*"')
                word_weight_pairs.append((word.strip('"'), float(weight)))
            except ValueError:
                continue

        if not word_weight_pairs:
            continue

        top_words, top_weights = zip(*word_weight_pairs[:8])
        ax.barh(top_words, top_weights, color=color_pool[t_id % len(color_pool)])
        ax.invert_yaxis()
        ax.set_title(f"Topic {t_id + 1}", fontsize=11)
        ax.set_xlabel("Probability Weight")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("LDA Topic Word Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig