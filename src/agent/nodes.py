import os
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_groq import ChatGroq


# ── LLM initialisation ────────────────────────────────────────────────────────

def _load_groq_client() -> ChatGroq:
    """
    Instantiate a Groq ChatGroq client.
    Reads the API key from Streamlit secrets first; falls back to the
    GROQ_API_KEY environment variable if secrets are unavailable.
    """
    try:
        import streamlit as st
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.getenv("GROQ_API_KEY", "")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=0.3)


# ── Node 1: Web search ────────────────────────────────────────────────────────

def search_node(state: dict) -> dict:
    """Query DuckDuckGo and store raw result objects in the pipeline state."""
    state["status"] = "Searching the web..."
    try:
        with DDGS() as engine:
            hits = list(engine.text(state["query"], max_results=6))
        state["search_results"] = hits
    except Exception as err:
        state["search_results"] = []
        state["error"] = f"Search failed: {err}"
    return state


# ── Node 2: Content retrieval ─────────────────────────────────────────────────

def retrieve_node(state: dict) -> dict:
    """
    Visit each URL from the search results and extract readable paragraph text.
    Falls back to the DuckDuckGo snippet when the page cannot be fetched.
    """
    state["status"] = "Retrieving source content..."
    fetched_pages = []

    for hit in state["search_results"]:
        url     = hit.get("href", "")
        title   = hit.get("title", "")
        snippet = hit.get("body", "")

        try:
            response = httpx.get(
                url,
                timeout          = 8,
                follow_redirects = True,
                headers          = {"User-Agent": "Mozilla/5.0"},
            )
            soup       = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            page_text  = " ".join(p.get_text() for p in paragraphs)[:3000]
            body       = page_text if len(page_text) > 200 else snippet
        except Exception:
            body = snippet

        fetched_pages.append({"url": url, "title": title, "text": body})

    state["retrieved_texts"] = fetched_pages
    return state


# ── Node 3: Source validation ─────────────────────────────────────────────────

def validate_node(state: dict) -> dict:
    """
    Filter out duplicate URLs and pages with insufficient textual content.
    Retains at most the five highest-quality sources.
    """
    state["status"] = "Validating sources..."
    seen_urls     = set()
    approved      = []

    for page in state["retrieved_texts"]:
        url     = page.get("url", "")
        content = page.get("text", "")
        if url not in seen_urls and len(content.strip()) > 100:
            approved.append(page)
            seen_urls.add(url)

    state["validated_sources"] = approved[:5]
    return state


# ── Node 4: LLM summarisation ─────────────────────────────────────────────────

def summarize_node(state: dict) -> dict:
    """
    Send combined source content to the Groq LLM and store
    a synthesised research summary back into the pipeline state.
    """
    state["status"] = "Summarizing with LLM..."
    try:
        llm = _load_groq_client()

        combined_text = "\n\n".join(
            f"Source: {src['title']}\n{src['text'][:1200]}"
            for src in state["validated_sources"]
        )

        prompt = (
            f'You are a research assistant. Using the web sources below, '
            f'write a comprehensive summary about:\n\n"{state["query"]}"\n\n'
            f"Sources:\n{combined_text}\n\n"
            f"Instructions:\n"
            f"- Write 3 to 5 clear, factual paragraphs\n"
            f"- Only use information drawn from the provided sources\n"
            f"- Do not introduce external knowledge or speculation\n"
            f"- Maintain a concise, academic tone"
        )

        llm_response        = llm.invoke(prompt)
        state["llm_summary"] = llm_response.content

    except Exception as err:
        state["llm_summary"] = "LLM summarisation failed. Verify your GROQ_API_KEY."
        state["error"]       = f"LLM error: {err}"

    return state


# ── Node 5: Report generation ─────────────────────────────────────────────────

def report_node(state: dict) -> dict:
    """
    Ask the LLM to structure the summary into a formatted research report.
    Parses the response into a dict with title, abstract, findings, and conclusion.
    """
    state["status"] = "Generating structured report..."

    try:
        llm = _load_groq_client()

        prompt = (
            f'Using the research summary below about "{state["query"]}", '
            f"produce a structured report in EXACTLY this format:\n\n"
            f"TITLE: [A descriptive research title]\n"
            f"ABSTRACT: [2–3 sentence overview]\n"
            f"KEY FINDINGS:\n"
            f"- [Finding 1]\n- [Finding 2]\n- [Finding 3]\n"
            f"- [Finding 4]\n- [Finding 5]\n"
            f"CONCLUSION: [2–3 sentence conclusion and implications]\n\n"
            f"Summary:\n{state['llm_summary']}"
        )

        response       = llm.invoke(prompt)
        state["report"] = _structure_report(
            response.content, state["query"], state["validated_sources"]
        )

    except Exception as err:
        state["report"] = {
            "title":        state["query"],
            "abstract":     state.get("llm_summary", ""),
            "key_findings": ["Report generation failed — see raw summary tab."],
            "conclusion":   "An error occurred during report generation.",
            "sources": [
                {"title": s.get("title", "Source"), "url": s.get("url", "")}
                for s in state["validated_sources"]
            ],
        }
        state["error"] = f"Report node error: {err}"

    state["status"] = "Complete"
    return state


# ── Internal: response parser ─────────────────────────────────────────────────

def _structure_report(raw_text: str, query: str, sources: list) -> dict:
    """
    Parse the LLM's labelled text output into a structured Python dict.
    Gracefully handles missing or partial sections.
    """
    parsed = {
        "title":        query,
        "abstract":     "",
        "key_findings": [],
        "conclusion":   "",
        "sources": [
            {"title": s.get("title", "Source"), "url": s.get("url", "")}
            for s in sources
        ],
    }
    active_section = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if line.startswith("TITLE:"):
            parsed["title"]   = line.replace("TITLE:", "").strip()
        elif line.startswith("ABSTRACT:"):
            parsed["abstract"] = line.replace("ABSTRACT:", "").strip()
            active_section     = "abstract"
        elif line.startswith("KEY FINDINGS:"):
            active_section = "findings"
        elif line.startswith("CONCLUSION:"):
            parsed["conclusion"] = line.replace("CONCLUSION:", "").strip()
            active_section       = "conclusion"
        elif active_section == "findings" and line.startswith("-"):
            parsed["key_findings"].append(line.lstrip("- ").strip())
        elif active_section == "abstract" and line and not any(
            line.startswith(k) for k in ["KEY FINDINGS", "CONCLUSION", "TITLE"]
        ):
            parsed["abstract"] += (" " + line) if parsed["abstract"] else line
        elif active_section == "conclusion" and line and not any(
            line.startswith(k) for k in ["KEY FINDINGS", "ABSTRACT", "TITLE"]
        ):
            parsed["conclusion"] += (" " + line) if parsed["conclusion"] else line

    return parsed
