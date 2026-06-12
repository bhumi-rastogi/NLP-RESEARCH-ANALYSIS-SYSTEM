def format_report(pipeline_state: dict) -> dict:
    """
    Extract and sanitise all report fields from the final pipeline state.
    Returns a clean dict that the Streamlit UI can render without crashing,
    even if the LLM or a node produced incomplete output.
    """
    raw       = pipeline_state.get("report", {})
    user_q    = pipeline_state.get("query", "Research Report")

    heading    = raw.get("title")      or user_q
    overview   = raw.get("abstract")   or "No abstract was generated."
    highlights = raw.get("key_findings") or []
    closing    = raw.get("conclusion") or "No conclusion was generated."
    refs       = raw.get("sources")    or []

    # Normalise each source to a guaranteed {title, url} dict
    clean_refs = []
    for entry in refs:
        if isinstance(entry, dict):
            clean_refs.append({
                "title": entry.get("title", "Untitled Source"),
                "url":   entry.get("url",   ""),
            })

    return {
        "title":        heading,
        "abstract":     overview,
        "key_findings": highlights,
        "conclusion":   closing,
        "sources":      clean_refs,
    }
