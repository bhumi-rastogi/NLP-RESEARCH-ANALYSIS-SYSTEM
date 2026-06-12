from langgraph.graph import StateGraph, END
from src.agent.state import PipelineState
from src.agent.nodes import (
    search_node,
    retrieve_node,
    validate_node,
    summarize_node,
    report_node,
)


def _assemble_pipeline():
    """
    Wire all five research nodes into a linear LangGraph workflow and
    return the compiled, executable graph.
    """
    workflow = StateGraph(PipelineState)

    # Register each processing node
    workflow.add_node("search",    search_node)
    workflow.add_node("retrieve",  retrieve_node)
    workflow.add_node("validate",  validate_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("report",    report_node)

    # Chain nodes sequentially: search → retrieve → validate → summarize → report → END
    workflow.set_entry_point("search")
    workflow.add_edge("search",    "retrieve")
    workflow.add_edge("retrieve",  "validate")
    workflow.add_edge("validate",  "summarize")
    workflow.add_edge("summarize", "report")
    workflow.add_edge("report",    END)

    return workflow.compile()


# Module-level compiled graph — imported directly by the Streamlit app
_pipeline = _assemble_pipeline()


def _make_initial_state(user_query: str) -> PipelineState:
    """Build a blank PipelineState seeded with the user's query."""
    return PipelineState(
        query             = user_query,
        search_results    = [],
        retrieved_texts   = [],
        validated_sources = [],
        llm_summary       = "",
        report            = {},
        status            = "Initialising...",
        error             = None,
    )


def run_research_agent(user_query: str) -> dict:
    """
    Execute the full research pipeline synchronously.
    Returns the final PipelineState after all nodes have run.
    """
    return _pipeline.invoke(_make_initial_state(user_query))


def stream_research_agent(user_query: str):
    """
    Execute the research pipeline with incremental streaming.
    Yields one state-update event per completed node so the UI can
    update its progress indicator in real time.
    """
    for event in _pipeline.stream(_make_initial_state(user_query)):
        yield event
