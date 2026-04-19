# app.py

# Streamlit web UI for the NLP Research Analyzer.
# Run with: streamlit run app.py

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import yaml
from wordcloud import WordCloud

# ensure src/ package is importable 
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlp_research.ingestion import ingest_files
from nlp_research.preprocessing import preprocess, split_into_paragraphs
from nlp_research.features import build_tfidf_matrix, get_top_keywords
from nlp_research.modeling import build_corpus, train_lda, compute_coherence, sweep_num_topics
from nlp_research.summarization import summarize

# load config
CFG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

# page setup
st.set_page_config(
    page_title="NLP Research Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 NLP Research Analyzer")
st.caption("Upload one or more research documents (PDF or TXT) to extract topics, keywords, and summaries.")

# sidebar controls 
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload documents (.pdf or .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    num_topics = st.slider(
        "Number of LDA Topics",
        min_value=2, max_value=10,
        value=CFG["lda"]["num_topics"],
    )

    top_n_summary = st.slider(
        "Summary Sentences",
        min_value=3, max_value=15,
        value=CFG["summarization"]["top_n"],
    )

    top_n_keywords = st.slider(
        "Top Keywords",
        min_value=5, max_value=30,
        value=CFG["tfidf"]["top_n_keywords"],
    )



    run = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

#  main logic
if not uploaded_files:
    st.info("👈 Upload at least one PDF or TXT file from the sidebar to get started.")
    st.stop()

if not run:
    st.info(f"📂 {len(uploaded_files)} file(s) loaded. Click **Run Analysis** in the sidebar when ready.")
    st.stop()

#  save uploads to temp files and ingest 
with st.status("⏳ Processing documents…", expanded=True) as status:
    tmp_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            tmp_path = Path(tmpdir) / uf.name
            tmp_path.write_bytes(uf.read())
            tmp_paths.append(tmp_path)

        st.write(f"📄 Ingesting {len(tmp_paths)} file(s)…")
        raw_text = ingest_files(tmp_paths)

    st.write(f"📝 Extracted **{len(raw_text):,}** characters total.")

    st.write("🧹 Preprocessing…")
    tokens = preprocess(raw_text, min_word_len=CFG["preprocessing"]["min_word_len"])
    # preprocess() returns a list of tokens
    if not tokens:
        st.error("Document is empty or contains only stop words after preprocessing. Please upload a valid document.")
        st.stop()
    processed_text = " ".join(tokens)  # string for TF-IDF and wordcloud

    st.write("📊 Computing TF-IDF…")
    tfidf_matrix, tfidf_vectorizer = build_tfidf_matrix(
        [processed_text],
        max_df=CFG["tfidf"]["max_df"],
        min_df=CFG["tfidf"]["min_df"],
    )
    top_keywords = get_top_keywords(tfidf_matrix, tfidf_vectorizer, top_n=top_n_keywords)

    st.write("🧠 Training LDA model…")
    paragraphs = split_into_paragraphs(raw_text, min_length=CFG["preprocessing"]["min_paragraph_len"])
    # If too few paragraphs, fall back to splitting the whole text into sentences
    if len(paragraphs) < 3:
        from nlp_research.preprocessing import split_into_sentences
        paragraphs = split_into_sentences(raw_text)
    # preprocess() returns a token list — gensim needs list-of-token-lists
    cleaned_docs = [preprocess(p) for p in paragraphs]
    # Filter out empty token lists
    cleaned_docs = [doc for doc in cleaned_docs if doc]
    if not cleaned_docs:
        st.warning("Not enough text to train LDA topics. Skipping topic modelling.")
        lda_model = None
        coherence = 0.0
        sweep = [(k, 0.0) for k in range(*CFG["lda"]["sweep_range"])]
    else:
        dictionary, corpus = build_corpus(cleaned_docs)
        # Cap num_topics to available unique tokens
        actual_topics = min(num_topics, len(dictionary))
        lda_model = train_lda(
            corpus, dictionary,
            num_topics=actual_topics,
            passes=CFG["lda"]["passes"],
            random_state=CFG["lda"]["random_state"],
        )
        coherence = compute_coherence(lda_model, cleaned_docs, dictionary)

    if cleaned_docs:
        st.write("📈 Computing coherence sweep…")
        sweep = sweep_num_topics(
            corpus, dictionary, cleaned_docs,
            topic_range=range(*CFG["lda"]["sweep_range"]),
            passes=CFG["lda"]["passes"],
        )

    st.write("📝 Generating summary…")
    summary = summarize(raw_text, top_n=top_n_summary)

    status.update(label="✅ Analysis complete!", state="complete", expanded=False)

#  results tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Keywords", "🧠 Topics", "📈 Coherence", "☁️ Word Cloud", "📝 Summary"
])

#  Tab 1: Keywords 
with tab1:
    st.subheader("Top Keywords (TF-IDF)")
    words, scores = zip(*top_keywords)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(list(words), list(scores), color="steelblue")
    ax.set_xlabel("TF-IDF Score")
    ax.set_title("Top Keywords")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Keyword Table")
    st.dataframe(
        {"Keyword": list(words), "TF-IDF Score": [round(s, 4) for s in scores]},
        use_container_width=True,
    )

#  Tab 2: LDA Topics 
with tab2:
    if lda_model is None:
        st.warning("Not enough text to train LDA topics. Upload a longer document.")
    else:
        st.subheader(f"LDA Topics  —  Coherence Score: `{coherence:.4f}`")
        cols = st.columns(min(num_topics, 3))
        for i, topic in lda_model.show_topics(num_words=8, formatted=False):
            t_words = [w for w, _ in topic]
            t_probs = [p for _, p in topic]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(t_words, t_probs, color="darkorange")
            ax.set_title(f"Topic {i}")
            ax.set_xticklabels(t_words, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("Probability")
            cols[i % len(cols)].pyplot(fig)
            plt.close(fig)

#  Tab 3: Coherence Curve 
with tab3:
    st.subheader("Coherence vs Number of Topics")
    if not sweep or all(s == 0.0 for _, s in sweep):
        st.warning("Coherence sweep unavailable — not enough text for topic modelling.")
    else:
        nums, coh_scores = zip(*sweep)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(nums), list(coh_scores), marker="o", color="mediumseagreen", linewidth=2)
        ax.set_xlabel("Number of Topics")
        ax.set_ylabel("Coherence Score (c_v)")
        ax.set_title("Coherence Curve")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

#  Tab 4: Word Cloud 
with tab4:
    st.subheader("Word Cloud")
    wc = WordCloud(
        width=CFG["visualization"]["wordcloud_width"],
        height=CFG["visualization"]["wordcloud_height"],
        background_color="white",
    ).generate(processed_text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

#  Tab 5: Summary 
with tab5:
    st.subheader(f"Extractive Summary — Top {top_n_summary} Sentences")
    for i, sentence in enumerate(summary, 1):
        st.markdown(f"**{i}.** {sentence}")
