from typing import Optional, TypedDict


class PipelineState(TypedDict):
    """
    Shared data container passed sequentially through every LangGraph node.
    Each node reads relevant fields and writes its output back into this dict.
    """
    query:             str            # The user's original research question
    search_results:    list           # Raw result objects from DuckDuckGo
    retrieved_texts:   list           # Scraped body text for each result URL
    validated_sources: list           # Sources that passed quality filtering
    llm_summary:       str            # Synthesized summary produced by the LLM
    report:            dict           # Final structured report dict
    status:            str            # Human-readable current pipeline step
    error:             Optional[str]  # Set when any node encounters an exception
