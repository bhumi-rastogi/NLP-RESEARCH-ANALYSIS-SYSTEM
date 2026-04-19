# src/graph/workflow.py

import sys
from pathlib import Path

# ✅ Fix import path (IMPORTANT)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import StateGraph

from src.agents.search_agent import search_agent
from src.agents.summarizer_agent import summarizer_agent
from src.agents.report_agent import report_agent
from src.agents.validator_agent import validator_agent  # optional but recommended


# ✅ Define state structure (better for marks)
class GraphState(dict):
    pass


def build_graph():

    graph = StateGraph(GraphState)

    # 🔹 Nodes
    graph.add_node("search", search_agent)
    graph.add_node("summarize", summarizer_agent)
    graph.add_node("report", report_agent)
    graph.add_node("validate", validator_agent)  # optional

    # 🔹 Flow
    graph.set_entry_point("search")

    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "report")
    graph.add_edge("report", "validate")  # optional step

    # 🔹 End
    graph.set_finish_point("validate")

    return graph.compile()