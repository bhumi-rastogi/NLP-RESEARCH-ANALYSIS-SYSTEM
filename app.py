import streamlit as st
import pandas as pd

# ── M1 imports ─────────────────────────────────────────────────────────────────
from src.preprocessing import preprocess_text, get_text_stats
from src.document_loader import load_uploaded_files
from src.feature_extraction import build_tfidf
from src.topic_model import build_lda_model
from src.evaluation import calculate_coherence
from src.keyword_extractor import extract_keywords
from src.summarizer import summarize_text
from src.visualizations import generate_wordcloud, plot_top_keywords, plot_topic_distribution

# ── M2 imports ─────────────────────────────────────────────────────────────────
from src.agent.graph import run_research_agent, stream_research_agent
from src.agent.report_generator import format_report
from src.pdf_export import generate_pdf_report

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusResearch · AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }
  .stApp {
    background: #050816;
  }

  /* ── Animated gradient background ── */
  .stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
      radial-gradient(ellipse at 20% 20%, rgba(124, 58, 237, 0.12) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 50% 50%, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(20px);
  }
  [data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
  }

  /* ── Hero Band ── */
  .hero-band {
    background: linear-gradient(135deg, rgba(124,58,237,0.18) 0%, rgba(16,185,129,0.12) 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .hero-band::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
  }
  .hero-title-m2 {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #34d399 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
  }
  .hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 400;
  }
  .hero-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1.2rem;
  }
  .hero-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Mode Toggle ── */
  div[data-testid="stRadio"] > div {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 10px 22px;
    color: #94a3b8 !important;
    font-size: 0.92rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s ease;
  }
  div[data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(167,139,250,0.4);
    color: #e2e8f0 !important;
  }
  div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(16,185,129,0.2));
    border-color: #7c3aed;
    color: #fff !important;
    font-weight: 700;
    box-shadow: 0 0 20px rgba(124,58,237,0.25);
  }

  /* ── Metric Cards ── */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.2rem !important;
    transition: border-color 0.2s;
  }
  [data-testid="stMetric"]:hover {
    border-color: rgba(167,139,250,0.35);
  }
  [data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.06) !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.5), rgba(16,185,129,0.3)) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.3) !important;
  }

  /* ── Buttons ── */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    letter-spacing: 0.02em;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.55) !important;
    filter: brightness(1.08) !important;
  }
  .stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
  }
  .stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
  }
  .stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(167,139,250,0.5) !important;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
  }
  .stTextInput > div > div > input::placeholder,
  .stTextArea > div > div > textarea::placeholder {
    color: #475569 !important;
  }

  /* ── File Uploader ── */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.5) !important;
  }

  /* ── Sliders ── */
  .stSlider [data-baseweb="slider"] {
    padding-top: 6px !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
  }
  .streamlit-expanderHeader:hover {
    background: rgba(255,255,255,0.07) !important;
  }
  .streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
  }

  /* ── Alerts ── */
  .stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    border-left-width: 4px !important;
  }
  .stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border-color: #10b981 !important;
    color: #d1fae5 !important;
  }
  .stInfo {
    background: rgba(59,130,246,0.1) !important;
    border-color: #3b82f6 !important;
    color: #dbeafe !important;
  }
  .stWarning {
    background: rgba(245,158,11,0.1) !important;
    border-color: #f59e0b !important;
    color: #fef3c7 !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08) !important;
  }

  /* ── Divider ── */
  hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 1.2rem 0 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 99px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

  /* ── Custom Components ── */
  .keyword-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(16,185,129,0.15));
    border: 1px solid rgba(167,139,250,0.3);
    color: #c4b5fd;
    border-radius: 8px;
    padding: 3px 12px;
    margin: 3px 4px;
    font-size: 0.82rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.2s;
  }
  .keyword-badge:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.4), rgba(16,185,129,0.25));
    border-color: rgba(167,139,250,0.6);
  }

  .finding-card {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 4px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1.2rem;
    margin: 6px 0;
    color: #d1fae5;
    font-size: 0.92rem;
    line-height: 1.55;
    transition: background 0.2s;
  }
  .finding-card:hover {
    background: rgba(16,185,129,0.1);
  }

  /* ── Progress Tracker ── */
  .progress-wrap {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
  }
  .progress-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .progress-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    font-size: 0.9rem;
  }
  .step-icon {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 700;
    flex-shrink: 0;
  }
  .step-icon.completed { background: rgba(16,185,129,0.2); border: 1.5px solid #10b981; color: #10b981; }
  .step-icon.active    { background: rgba(59,130,246,0.2);  border: 1.5px solid #60a5fa; color: #60a5fa; }
  .step-icon.pending   { background: rgba(255,255,255,0.04); border: 1.5px solid rgba(255,255,255,0.12); color: #475569; }
  .step-label { font-weight: 600; }
  .step-label.completed { color: #34d399; }
  .step-label.active    { color: #93c5fd; }
  .step-label.pending   { color: #475569; }

  /* ── Source Card ── */
  .source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin: 6px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s, background 0.2s;
  }
  .source-card:hover {
    border-color: rgba(96,165,250,0.35);
    background: rgba(59,130,246,0.05);
  }
  .source-num {
    min-width: 28px; height: 28px;
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; color: #60a5fa;
  }

  /* ── Section Header ── */
  .section-head {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 1.4rem 0 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);
    margin-left: 8px;
  }

  /* ── Info chips in sidebar ── */
  .sidebar-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 0;
  }

  /* hide default radio label ── */
  div[data-testid="stRadio"] > label {
    display: none !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<p style='font-size:0.78rem;color:#475569;text-transform:uppercase;"
    "letter-spacing:0.1em;font-weight:600;margin-bottom:6px;'>SELECT MODE</p>",
    unsafe_allow_html=True
)
mode = st.radio(
    label="Select Mode",
    options=["🔬 Milestone 1 — Classical NLP", "🤖 Milestone 2 — Agentic AI"],
    horizontal=True,
    key="mode_toggle",
    label_visibility="collapsed",
    index=1,
)
st.markdown("<hr style='margin:1rem 0 1.5rem'/>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 1
# ══════════════════════════════════════════════════════════════════════════════
if "Milestone 1" in mode:

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='padding:0.5rem 0 1rem'>
          <p style='font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#34d399);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
            NexusResearch
          </p>
          <p style='font-size:0.8rem;color:#64748b;margin:2px 0 0'>Classical NLP Pipeline</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        input_mode = st.radio(
            "Input Mode",
            ["📝 Paste Text", "📂 Upload Documents"],
            help="Choose how to provide your research content."
        )
        st.markdown("---")

        st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;"
                    "letter-spacing:0.08em;font-weight:600'>Model Settings</p>", unsafe_allow_html=True)
        num_topics = st.slider("LDA Topics", min_value=2, max_value=10, value=5)
        num_keywords = st.slider("Top Keywords", min_value=5, max_value=30, value=15)
        num_summary_sentences = st.slider("Summary Sentences", min_value=2, max_value=10, value=5)
        st.markdown("---")

        st.markdown("""
        <div style='background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
          border-radius:12px;padding:0.9rem 1rem;font-size:0.82rem;color:#94a3b8;line-height:1.6'>
          <b style='color:#a78bfa'>About M1</b><br>
          Traditional NLP pipeline using TF-IDF, LDA topic modelling, and extractive summarization.
          No LLMs or external APIs required.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:1rem;display:flex;flex-direction:column;gap:4px'>
          <span class='sidebar-chip'>⚙ TF-IDF + LDA</span>
          <span class='sidebar-chip'>📊 Extractive Summary</span>
          <span class='sidebar-chip'>🔒 100% Offline</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-band'>
      <p class='hero-title'>🔬 NexusResearch</p>
      <p class='hero-sub'>Milestone 1 — Intelligent Research Topic Analyzer · Classical NLP Pipeline</p>
      <div class='hero-pills'>
        <span class='hero-pill'>TF-IDF</span>
        <span class='hero-pill'>LDA Topic Modelling</span>
        <span class='hero-pill'>Extractive Summarization</span>
        <span class='hero-pill'>Word Cloud</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Section ────────────────────────────────────────────────────────
    raw_text = ""
    doc_names = []

    if "Paste Text" in input_mode:
        raw_text = st.text_area(
            "✍️ Paste your research text below",
            height=180,
            placeholder="Paste any research paper abstract, article, report, or multi-paragraph text here…"
        )
        if raw_text.strip():
            doc_names = ["Pasted Text"]
    else:
        uploaded_files = st.file_uploader(
            "📂 Upload research documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload one or more PDF or TXT documents."
        )
        if uploaded_files:
            with st.spinner("Reading documents…"):
                docs = load_uploaded_files(uploaded_files)
            if docs:
                raw_text = "\n\n".join(d["text"] for d in docs)
                doc_names = [d["name"] for d in docs]
                st.success(f"✅ Loaded **{len(docs)}** document(s): {', '.join(doc_names)}")
            else:
                st.error("Could not extract text from the uploaded files.")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_analysis = st.button("🚀 Analyze Research", type="primary", use_container_width=True)

    # ── Results ──────────────────────────────────────────────────────────────
    if run_analysis:
        if not raw_text.strip():
            st.warning("⚠️ Please provide some text or upload a document first.")
        else:
            with st.spinner("Running NLP pipeline…"):
                stats = get_text_stats(raw_text)
                processed = preprocess_text(raw_text)
                corpus = [processed]
                tfidf_matrix, feature_names, _ = build_tfidf(corpus)
                keywords = extract_keywords(tfidf_matrix, feature_names, top_n=num_keywords)
                lda_model, topics, corpus_g, dictionary = build_lda_model(corpus, num_topics=num_topics)
                try:
                    coherence = calculate_coherence(lda_model, corpus, dictionary)
                except Exception:
                    coherence = None
                summary, scored_sentences = summarize_text(raw_text, num_sentences=num_summary_sentences)

            st.markdown("<hr/>", unsafe_allow_html=True)

            # ── Metrics Row ──────────────────────────────────────────────────
            st.markdown("<div class='section-head'>📊 Document Overview</div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Documents", len(doc_names))
            col2.metric("🔤 Words", f"{stats['words']:,}")
            col3.metric("📃 Sentences", stats["sentences"])
            col4.metric("🔠 Unique Tokens", stats["unique_tokens"])

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── Tabs ─────────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔑 Keywords", "🧩 Topics", "📝 Summary", "📈 Visualizations"
            ])

            with tab1:
                st.markdown("<div class='section-head'>Top Keywords by TF-IDF Score</div>", unsafe_allow_html=True)
                st.caption("Words ranked by term frequency–inverse document frequency weight.")
                badge_html = "".join(
                    f'<span class="keyword-badge">{w} <b style="opacity:.7">({s:.3f})</b></span>'
                    for w, s in keywords
                )
                st.markdown(f"<div style='margin:1rem 0'>{badge_html}</div>", unsafe_allow_html=True)
                st.markdown("---")
                df_kw = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
                df_kw["TF-IDF Score"] = df_kw["TF-IDF Score"].round(4)
                df_kw.index += 1
                st.dataframe(df_kw, use_container_width=True)

            with tab2:
                st.markdown("<div class='section-head'>LDA Topic Clusters</div>", unsafe_allow_html=True)
                if coherence is not None:
                    coh_col, _ = st.columns([1, 3])
                    coh_col.metric(
                        "Coherence Score (Cᵥ)", f"{coherence:.4f}",
                        help="Higher is better (0.4–0.7 = good topic separation)."
                    )
                st.caption("Each topic is represented by its highest-probability words.")
                st.markdown("---")
                for topic_id, topic_str in topics:
                    with st.expander(f"🔹 Topic {topic_id + 1}", expanded=(topic_id < 2)):
                        pairs = []
                        for part in topic_str.split(" + "):
                            try:
                                weight, word = part.split('*"')
                                pairs.append((word.strip('"'), float(weight)))
                            except ValueError:
                                continue
                        badges = "".join(
                            f'<span class="keyword-badge">{w} <b style="opacity:.7">({wt:.3f})</b></span>'
                            for w, wt in pairs
                        )
                        st.markdown(f"<div style='padding:0.5rem 0'>{badges}</div>", unsafe_allow_html=True)

            with tab3:
                st.markdown("<div class='section-head'>Extractive Summary</div>", unsafe_allow_html=True)
                st.caption("Sentences selected by TF-IDF score, in original document order.")
                st.markdown("---")
                if summary:
                    st.markdown(
                        f"<div style='background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.2);"
                        f"border-left:4px solid #10b981;border-radius:0 12px 12px 0;padding:1.2rem 1.4rem;"
                        f"color:#d1fae5;font-size:0.95rem;line-height:1.7'>{summary}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Not enough text to generate a summary.")
                if scored_sentences:
                    st.markdown("<div class='section-head' style='margin-top:1.5rem'>Sentence Scores</div>",
                                unsafe_allow_html=True)
                    df_sum = pd.DataFrame(scored_sentences, columns=["Sentence", "Score"])
                    df_sum["Score"] = df_sum["Score"].round(4)
                    df_sum.index += 1
                    st.dataframe(df_sum, use_container_width=True)

            with tab4:
                st.markdown("<div class='section-head'>Analytical Visualizations</div>", unsafe_allow_html=True)
                v1, v2 = st.columns(2)
                with v1:
                    st.markdown("<p style='font-weight:600;color:#94a3b8;font-size:0.9rem'>☁️ Word Cloud</p>",
                                unsafe_allow_html=True)
                    if processed.strip():
                        st.pyplot(generate_wordcloud(processed))
                    else:
                        st.info("Not enough text for a word cloud.")
                with v2:
                    st.markdown("<p style='font-weight:600;color:#94a3b8;font-size:0.9rem'>📊 Keyword Bar Chart</p>",
                                unsafe_allow_html=True)
                    if keywords:
                        st.pyplot(plot_top_keywords(keywords))
                st.markdown("---")
                st.markdown("<div class='section-head'>Topic Word Distribution</div>", unsafe_allow_html=True)
                topic_fig = plot_topic_distribution(topics)
                if topic_fig:
                    st.pyplot(topic_fig)


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 2
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 2" in mode:

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='padding:0.5rem 0 1rem'>
          <p style='font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#34d399,#60a5fa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
            NexusResearch
          </p>
          <p style='font-size:0.8rem;color:#64748b;margin:2px 0 0'>Agentic AI Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;"
                    "letter-spacing:0.08em;font-weight:600'>How It Works</p>", unsafe_allow_html=True)

        steps_info = [
            ("🔎", "Web Search", "DuckDuckGo (no key)"),
            ("📥", "Retrieve Content", "HTTP scraping"),
            ("✅", "Validate Sources", "Quality filter"),
            ("🧠", "LLM Summarize", "Groq · Llama 3.3"),
            ("📝", "Structured Report", "Parsed + formatted"),
        ]
        for icon, title, sub in steps_info:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:6px 0;
              background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
              border-radius:10px;padding:8px 10px'>
              <span style='font-size:1.1rem'>{icon}</span>
              <div>
                <p style='margin:0;font-size:0.85rem;font-weight:600;color:#e2e8f0'>{title}</p>
                <p style='margin:0;font-size:0.72rem;color:#475569;font-family:"JetBrains Mono",monospace'>{sub}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='display:flex;flex-direction:column;gap:4px'>
          <span class='sidebar-chip'>🤖 LLM: llama-3.3-70b</span>
          <span class='sidebar-chip'>🔍 Search: DuckDuckGo</span>
          <span class='sidebar-chip'>📄 Export: PDF</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-band' style='background:linear-gradient(135deg,rgba(16,185,129,0.14) 0%,rgba(59,130,246,0.10) 100%);
      border-color:rgba(52,211,153,0.25)'>
      <div style='position:absolute;top:-60px;right:-60px;width:220px;height:220px;
        background:radial-gradient(circle,rgba(52,211,153,0.2) 0%,transparent 70%);border-radius:50%;pointer-events:none'></div>
      <p class='hero-title-m2'>🤖 Agentic Research Assistant</p>
      <p class='hero-sub'>Milestone 2 — LangGraph · Web Search · LLM Summarization · Structured Reports</p>
      <div class='hero-pills'>
        <span class='hero-pill'>LangGraph</span>
        <span class='hero-pill'>DuckDuckGo Search</span>
        <span class='hero-pill'>Groq LLM</span>
        <span class='hero-pill'>PDF Export</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query Input ──────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Research Query",
        placeholder="e.g. What are the latest advances in transformer-based NLP models?",
        label_visibility="visible",
    )

    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        run_agent = st.button("🚀 Start Research", type="primary", use_container_width=True)

    # ── Agent Execution ───────────────────────────────────────────────────────
    if run_agent:
        if not query.strip():
            st.warning("⚠️ Please enter a research query first.")
        else:
            final_state = None
            ui_placeholder = st.empty()

            steps_order = ["search", "retrieve", "validate", "summarize", "report"]
            step_status = {step: "pending" for step in steps_order}
            step_status["search"] = "active"

            step_meta = {
                "search":    ("🔎", "Searching the web",         "Search complete"),
                "retrieve":  ("📥", "Retrieving source content",  "Content retrieved"),
                "validate":  ("✅", "Validating sources",         "Sources validated"),
                "summarize": ("🧠", "Summarizing with LLM",       "Summary generated"),
                "report":    ("📝", "Generating structured report","Report ready"),
            }

            def render_progress(is_complete=False, has_error=False):
                if has_error:
                    header = "⚠️ Completed with warnings"
                    header_color = "#f59e0b"
                elif is_complete:
                    header = "✅ Research complete!"
                    header_color = "#10b981"
                else:
                    header = "⏳ Running Research Agent…"
                    header_color = "#60a5fa"

                rows = ""
                for i, step in enumerate(steps_order, 1):
                    status = step_status[step]
                    icon_char, active_label, done_label = step_meta[step]

                    if status == "completed":
                        dot_style = "background:#10b981;border:2px solid #34d399;"
                        label_color = "#34d399"
                        label_text = done_label
                        dot_content = "✓"
                    elif status == "active":
                        dot_style = "background:rgba(96,165,250,0.2);border:2px solid #60a5fa;"
                        label_color = "#93c5fd"
                        label_text = active_label + "…"
                        dot_content = icon_char
                    else:
                        dot_style = "background:rgba(255,255,255,0.03);border:2px solid rgba(255,255,255,0.1);"
                        label_color = "#334155"
                        label_text = active_label
                        dot_content = str(i)

                    rows += f"""
                    <div style='display:flex;align-items:center;gap:14px;padding:7px 0;'>
                      <div style='width:32px;height:32px;border-radius:50%;{dot_style}
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.85rem;font-weight:700;color:{label_color};flex-shrink:0;'>
                        {dot_content}
                      </div>
                      <div>
                        <p style='margin:0;font-size:0.88rem;font-weight:600;color:{label_color}'>{label_text}</p>
                        <p style='margin:0;font-size:0.72rem;color:#334155;font-family:"JetBrains Mono",monospace'>
                          Step {i} of 5
                        </p>
                      </div>
                    </div>
                    """

                html = f"""
                <div style='background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.09);
                  border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1rem;backdrop-filter:blur(12px);'>
                  <p style='font-size:1rem;font-weight:700;color:{header_color};margin:0 0 1rem;
                    display:flex;align-items:center;gap:8px;'>{header}</p>
                  {rows}
                </div>
                """
                ui_placeholder.markdown(html, unsafe_allow_html=True)

            render_progress()

            for event in stream_research_agent(query):
                for node_name, state in event.items():
                    final_state = state
                    if node_name in step_status:
                        step_status[node_name] = "completed"
                        curr_idx = steps_order.index(node_name)
                        if curr_idx + 1 < len(steps_order):
                            step_status[steps_order[curr_idx + 1]] = "active"
                render_progress()

            has_error = bool(final_state and final_state.get("error"))
            render_progress(is_complete=True, has_error=has_error)

            if has_error:
                st.warning(f"⚠️ Note: {final_state['error']}")

            report = format_report(final_state)

            # ── Result Tabs ──────────────────────────────────────────────────
            r_tab1, r_tab2, r_tab3 = st.tabs([
                "📄 Research Report", "🔗 Sources", "🧠 Raw LLM Output"
            ])

            with r_tab1:
                # Title
                st.markdown(
                    f"<h2 style='color:#f1f5f9;font-weight:800;font-size:1.7rem;"
                    f"margin-bottom:0.2rem'>{report['title']}</h2>",
                    unsafe_allow_html=True
                )
                st.markdown("<hr/>", unsafe_allow_html=True)

                # Abstract
                st.markdown("<div class='section-head'>📋 Abstract</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.2);"
                    f"border-left:4px solid #3b82f6;border-radius:0 12px 12px 0;padding:1.1rem 1.4rem;"
                    f"color:#dbeafe;font-size:0.94rem;line-height:1.7'>{report['abstract']}</div>",
                    unsafe_allow_html=True
                )

                # Key Findings
                st.markdown("<div class='section-head'>🔍 Key Findings</div>", unsafe_allow_html=True)
                findings = report["key_findings"]
                if findings:
                    for finding in findings:
                        st.markdown(f"<div class='finding-card'>✦ {finding}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No key findings were extracted.")

                # Conclusion
                st.markdown("<div class='section-head'>🏁 Conclusion</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.2);"
                    f"border-left:4px solid #10b981;border-radius:0 12px 12px 0;padding:1.1rem 1.4rem;"
                    f"color:#d1fae5;font-size:0.94rem;line-height:1.7'>{report['conclusion']}</div>",
                    unsafe_allow_html=True
                )

                # PDF Export
                st.markdown("<hr/>", unsafe_allow_html=True)
                try:
                    pdf_bytes = generate_pdf_report(report)
                    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                    with col_dl2:
                        st.download_button(
                            label="📥 Download Full Report as PDF",
                            data=pdf_bytes,
                            file_name=f"NexusResearch_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

            with r_tab2:
                st.markdown("<div class='section-head'>🔗 Validated Sources</div>", unsafe_allow_html=True)
                sources = report["sources"]
                if sources:
                    for i, source in enumerate(sources, 1):
                        title = source.get("title", "Unknown Source")
                        url = source.get("url", "")
                        link_html = (f'<a href="{url}" target="_blank" '
                                     f'style="color:#60a5fa;font-size:0.8rem;text-decoration:none;'
                                     f'font-family:\'JetBrains Mono\',monospace">Open ↗</a>'
                                     if url else "")
                        st.markdown(
                            f"""<div class='source-card'>
                              <div class='source-num'>{i}</div>
                              <div style='flex:1'>
                                <p style='margin:0;font-weight:600;color:#e2e8f0;font-size:0.9rem'>{title}</p>
                                {link_html}
                              </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No validated sources found.")

            with r_tab3:
                st.markdown("<div class='section-head'>🧠 Raw LLM Summary</div>", unsafe_allow_html=True)
                st.caption("Intermediate summary generated by the LLM before structuring the report.")
                raw_summary = final_state.get("llm_summary", "")
                if raw_summary:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);"
                        f"border-radius:12px;padding:1.2rem 1.4rem;color:#cbd5e1;font-size:0.9rem;"
                        f"line-height:1.75;white-space:pre-wrap'>{raw_summary}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No LLM summary available.")