# ResearchScope – Intelligent Research Analyzer
### From Classical NLP to Agentic AI Research Assistant

**🔗 Live App:** https://researchscopegenai.streamlit.app/

---

## Project Overview

ResearchScope is a dual-mode intelligent research analysis system built for the Generative AI course end-semester project. The application seamlessly integrates two complete research paradigms, switchable via a toggle button in the UI:

- **Milestone 1 (Classical NLP):** A fully offline, traditional NLP pipeline using TF-IDF, LDA Topic Modeling, and Extractive Summarization — no LLMs, no API calls.
- **Milestone 2 (Agentic AI):** A LangGraph-powered autonomous research agent that performs live web search, multi-source retrieval, LLM-based summarization, structured report generation, and PDF export.

---

## Constraints & Requirements

| Field | Details |
|---|---|
| **Team Members** | Kushal Sarkar, Chinmay Soni, Lakshya Bapna |
| **API Budget** | Free Tier Only (Groq, DuckDuckGo, Streamlit Cloud) |
| **LLM** | Groq — `llama-3.3-70b-versatile` (free tier) |
| **Hosting** | Streamlit Cloud |

---

## Technology Stack

### Milestone 1 — Classical NLP
| Component | Technology |
|---|---|
| Text Preprocessing | NLTK, spaCy (`en_core_web_sm`) |
| Feature Extraction | Scikit-learn (TfidfVectorizer) |
| Topic Modeling | Gensim (LDA, CoherenceModel) |
| Summarization | NLTK + Scikit-learn (Extractive) |
| Visualizations | Matplotlib, WordCloud |
| Document Loading | PyPDF2, Python I/O |

### Milestone 2 — Agentic AI
| Component | Technology |
|---|---|
| Agent Workflow | LangGraph (StateGraph, nodes, edges) |
| LLM Integration | Groq API — `llama-3.3-70b-versatile` |
| Web Search | DDGS (DuckDuckGo Search, no API key) |
| Web Retrieval | httpx + BeautifulSoup4 |
| PDF Export | ReportLab |
| State Management | TypedDict (ResearchState) |

### Shared
| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Language | Python 3.10+ |

---

## Milestones & Deliverables

### ✅ Milestone 1: Classical NLP Research Analysis System — COMPLETE

**Objective:** Build a robust baseline system using purely statistical and classical ML methods.

**Delivered:**
- **Document Intake:** Paste text directly or upload `.pdf` / `.txt` files (multi-file support via PyPDF2)
- **Dynamic Metrics Row:** Live stats — document count, word count, sentence count, unique token count
- **Keyword Extraction:** Top-N keywords ranked by TF-IDF score with badge display and sortable table
- **Topic Modeling:** LDA clusters with Coherence Score (Cᵥ) shown per topic in expandable sections
- **Extractive Summarization:** Sentences scored by TF-IDF, top-N selected in original document order
- **Visualizations:** Word Cloud, Keyword Bar Chart, Topic Word Distribution Chart
- **Streamlit UI:** Dark-mode interface with sidebar controls (topic count, keyword count, summary length sliders)

---

### ✅ Milestone 2: Agentic AI Research Assistant — COMPLETE

**Objective:** Transform the system into an autonomous agent that retrieves, reasons, and generates structured research reports.

**Delivered:**
- **5-Node LangGraph Workflow:**
  1. `search_node` — DuckDuckGo live web search (no API key needed)
  2. `retrieve_node` — Fetches full page content from each result URL via httpx + BeautifulSoup
  3. `validate_node` — Filters empty, duplicate, or low-quality sources; keeps top 5
  4. `summarize_node` — Groq LLM synthesizes a multi-paragraph research summary from sources
  5. `report_node` — LLM generates a structured report: Title, Abstract, Key Findings, Conclusion
- **Shared State:** `ResearchState` TypedDict maintains state across all 5 nodes
- **Robustness:** API failure fallbacks at every node — app never crashes on LLM errors
- **Structured Output:** Validated report with source URLs and clear attribution
- **Live Status Indicators:** Real-time step-by-step progress shown in the UI using `st.status()`
- **PDF Export (Extension):** Download the full research report as a professionally styled PDF via ReportLab
- **M1/M2 Toggle:** Single switch button in the app header toggles between both modes — M1 code is completely untouched

---

## System Architecture

### Milestone 1 Pipeline
```
Input (Text / PDF / TXT)
      ↓
Document Loader           [src/document_loader.py]
      ↓
Text Preprocessing        [src/preprocessing.py]
  - Lowercasing, Punctuation Removal
  - Tokenization, Stopword Removal, Lemmatization (spaCy)
      ↓
TF-IDF Feature Extraction [src/feature_extraction.py]
      ↓
      ├──→ Keyword Extraction    [src/keyword_extractor.py]
      └──→ LDA Topic Modeling    [src/topic_model.py]
              └──→ Coherence Score [src/evaluation.py]
      ↓
Extractive Summarization  [src/summarizer.py]
      ↓
UI & Visualizations       [app.py, src/visualizations.py]
```

### Milestone 2 — LangGraph Agent Pipeline
```
User Research Query (text input)
      ↓
Node 1: Search        [src/agent/nodes.py] — DuckDuckGo top 6 results
      ↓
Node 2: Retrieve      [src/agent/nodes.py] — Fetch & parse full page content
      ↓
Node 3: Validate      [src/agent/nodes.py] — Filter & deduplicate sources
      ↓
Node 4: Summarize     [src/agent/nodes.py] — Groq LLM evidence synthesis
      ↓
Node 5: Report        [src/agent/nodes.py] — Structured report generation
      ↓
Report Formatter      [src/agent/report_generator.py]
      ↓
UI Display + PDF Export [app.py, src/pdf_export.py]
```

---

## How to Run Locally

**1. Clone the Repository:**
```bash
git clone https://github.com/Kushal425/GENai_SecA_P1.git
cd GENai_SecA_P1
```

**2. Create and Activate a Virtual Environment:**

*macOS / Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```

*Windows:*
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download spaCy Language Model:**
```bash
python -m spacy download en_core_web_sm
```

**5. Download NLTK Data:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

**6. Set Up Groq API Key (required for Milestone 2 only):**

Create a file at `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free API key at [console.groq.com](https://console.groq.com) (no credit card required).

> Milestone 1 works completely offline — no API key needed.

**7. Launch the App:**
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## Project Structure

```
ResearchScope/
│
├── .streamlit/
│   └── secrets.toml              # Local API key (gitignored)
│
├── src/
│   ├── agent/                    # Milestone 2 — Agentic AI modules
│   │   ├── __init__.py
│   │   ├── state.py              # ResearchState TypedDict
│   │   ├── nodes.py              # 5 LangGraph node functions
│   │   ├── graph.py              # LangGraph StateGraph + run_research_agent()
│   │   └── report_generator.py  # Formats final state into clean report dict
│   │
│   ├── preprocessing.py          # Text cleaning, tokenization, lemmatization
│   ├── feature_extraction.py     # TF-IDF vectorization
│   ├── topic_model.py            # LDA topic modeling
│   ├── evaluation.py             # Coherence score evaluation
│   ├── keyword_extractor.py      # Keyword extraction logic
│   ├── summarizer.py             # Extractive summarization
│   ├── document_loader.py        # PDF and TXT file loading
│   ├── visualizations.py         # Word cloud, bar charts
│   └── pdf_export.py             # ReportLab PDF generation (M2 extension)
│
├── submission_docs/              # Report, architecture diagram
├── app.py                        # Main Streamlit app (M1 + M2 + toggle)
├── requirements.txt              # All Python dependencies
├── setup.sh                      # One-command setup script (macOS/Linux)
└── README.md                     # Project documentation
```

---

## Evaluation Criteria

| Phase | Weight | Criteria |
|---|---|---|
| **Milestone 1** | 25% | Preprocessing Quality, TF-IDF Feature Engineering, LDA Coherence Score, UI Functional Usability |
| **Milestone 2** | 30% | Agentic Reasoning Quality, Workflow Implementation, Structured Report Quality, UI & Deployment, Extension (PDF Export) |