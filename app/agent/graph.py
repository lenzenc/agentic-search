from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes.analyze import analyze_query
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.grade import grade_documents
from app.agent.nodes.rewrite import rewrite_query
from app.agent.nodes.generate import generate_answer
from app.agent.conditionals import route_after_analyze, route_after_grade


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges("analyze_query", route_after_analyze, {"retrieve": "retrieve"})
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
