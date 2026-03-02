# NLP Research Analyzer — Milestone 1

> **Course Project · Intelligent Research Topic Analysis & Agentic AI Research Assistant**
> Milestone 1: Traditional NLP pipeline — *no LLMs, no agentic workflows.*

---

## Features

| Capability | Technique |
|---|---|
| PDF text extraction | `PyPDF2` |
| Tokenization, stop-word removal, lemmatization | `NLTK` |
| Keyword extraction | TF-IDF (`scikit-learn`) |
| Topic modeling | LDA (`gensim`) |
| Coherence evaluation | `c_v` coherence (`gensim`) |
| Extractive summarization | TF-IDF sentence scoring |
| Visualizations | `matplotlib`, `wordcloud` |

---

## Project Structure

```
nlp_research_analyzer/
├── main.py                          ← pipeline entry point (CLI)
├── setup.py                         ← installable package
├── requirements.txt
├── config/
│   └── config.yaml                  ← all tunable parameters
├── data/
│   ├── raw/                         ← drop PDF(s) here
│   └── processed/                   ← (future) cached artefacts
├── outputs/
│   └── figures/                     ← saved plots
├── notebooks/
│   └── research_analysis.ipynb      ← original Colab notebook
└── src/
    └── nlp_research/
        ├── ingestion/               ← pdf_extractor.py
        ├── preprocessing/           ← text_preprocessor.py
        ├── features/                ← tfidf_extractor.py
        ├── modeling/                ← lda_topic_model.py
        ├── summarization/           ← extractive_summarizer.py
        └── visualization/           ← plotter.py
```

---

## Quick Start

### 1 — Install dependencies

```bash
cd nlp_research_analyzer
pip install -e .
```

> Or without editable install: `pip install -r requirements.txt`

### 2 — Add your PDF

```bash
cp /path/to/your/paper.pdf data/raw/
```

### 3 — Run the pipeline

```bash
# Display plots interactively
python main.py --pdf data/raw/paper.pdf

# Save plots to outputs/figures/ instead
python main.py --pdf data/raw/paper.pdf --save-plots

# Override defaults
python main.py --pdf data/raw/paper.pdf --num-topics 4 --top-n 7

# Skip visualizations entirely
python main.py --pdf data/raw/paper.pdf --no-plots
```

---

## Configuration

Edit `config/config.yaml` to change any parameter without touching source code:

```yaml
lda:
  num_topics: 3    # ← change this
  passes: 10

summarization:
  top_n: 5         # ← and this
```

---

## Pipeline Steps

```
PDF → [Ingestion] → raw text
    → [Preprocessing] → clean tokens
    → [TF-IDF] → keyword scores
    → [LDA] → topics + coherence
    → [Summarization] → extractive summary
    → [Visualization] → charts + word cloud
```

---

## Module API (quick reference)

```python
from nlp_research.ingestion import extract_text_from_pdf
from nlp_research.preprocessing import preprocess, split_into_paragraphs
from nlp_research.features import build_tfidf_matrix, get_top_keywords
from nlp_research.modeling import build_corpus, train_lda, compute_coherence
from nlp_research.summarization import summarize
from nlp_research.visualization import plot_top_keywords, plot_wordcloud
```

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for full dependency list
