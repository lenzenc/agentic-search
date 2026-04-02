"""Unit tests for LangGraph agent components. No live ES or LLM calls required."""
import pytest
from unittest.mock import AsyncMock, patch

from app.agent.conditionals import route_after_grade, route_after_analyze
from app.agent.state import AgentState, GradedDocument


def make_state(**overrides) -> AgentState:
    base: AgentState = {
        "original_query": "test query",
        "is_complex": False,
        "sub_queries": [],
        "active_query": "test query",
        "retrieved_docs": [],
        "iteration_count": 1,
        "graded_docs": [],
        "relevant_docs": [],
        "final_answer": "",
        "final_cards": [],
        "trajectory": [],
    }
    base.update(overrides)
    return base


# ── route_after_analyze ────────────────────────────────────────────────────────

def test_route_after_analyze_always_retrieve():
    state = make_state(is_complex=False)
    assert route_after_analyze(state) == "retrieve"

def test_route_after_analyze_complex_still_retrieve():
    state = make_state(is_complex=True)
    assert route_after_analyze(state) == "retrieve"


# ── route_after_grade ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("MIN_RELEVANT_DOCS", "3")
    monkeypatch.setenv("MAX_RETRIEVE_ITERATIONS", "3")


def make_relevant_docs(n: int) -> list[dict]:
    return [{"card_id": str(i), "name": f"Card {i}"} for i in range(n)]


def test_route_enough_relevant_docs_goes_to_generate():
    state = make_state(relevant_docs=make_relevant_docs(3), iteration_count=1)
    assert route_after_grade(state) == "generate_answer"


def test_route_more_than_enough_relevant_docs():
    state = make_state(relevant_docs=make_relevant_docs(5), iteration_count=1)
    assert route_after_grade(state) == "generate_answer"


def test_route_too_few_relevant_under_max_iter_rewrites():
    state = make_state(relevant_docs=make_relevant_docs(2), iteration_count=1)
    assert route_after_grade(state) == "rewrite_query"


def test_route_zero_relevant_under_max_iter_rewrites():
    state = make_state(relevant_docs=[], iteration_count=2)
    assert route_after_grade(state) == "rewrite_query"


def test_route_max_iterations_reached_generates_anyway():
    # Even with 0 relevant docs, hitting max iterations forces generation
    state = make_state(relevant_docs=[], iteration_count=3)
    assert route_after_grade(state) == "generate_answer"


def test_route_over_max_iterations_generates():
    state = make_state(relevant_docs=[], iteration_count=4)
    assert route_after_grade(state) == "generate_answer"


# ── analyze_query node (mocked Claude) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_simple_query():
    simple_response = '{"is_complex": false, "sub_queries": ["fire type pokemon"]}'

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text=simple_response)]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    with patch("app.agent.nodes.analyze.get_client", return_value=mock_client):
        from app.agent.nodes.analyze import analyze_query
        state = make_state(original_query="fire type pokemon")
        result = await analyze_query(state)

    assert result["is_complex"] is False
    assert len(result["sub_queries"]) == 1
    assert result["sub_queries"][0]["query"] == "fire type pokemon"
    assert len(result["trajectory"]) == 1
    assert result["trajectory"][0]["node"] == "analyze_query"


@pytest.mark.asyncio
async def test_analyze_complex_query():
    complex_response = '{"is_complex": true, "sub_queries": ["electric type pokemon", "pokemon ability causes paralysis"]}'

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text=complex_response)]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    with patch("app.agent.nodes.analyze.get_client", return_value=mock_client):
        from app.agent.nodes.analyze import analyze_query
        state = make_state(original_query="electric pokemon that paralyze")
        result = await analyze_query(state)

    assert result["is_complex"] is True
    assert len(result["sub_queries"]) == 2
    assert result["trajectory"][0]["metadata"]["is_complex"] is True


# ── grade_documents node (mocked Claude) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_grade_documents_relevant():
    grade_response = '{"grade": "relevant", "reasoning": "Card has fire type with high HP"}'

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text=grade_response)]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    card = {"card_id": "test-1", "name": "Charizard", "types": ["Fire"], "hp": 330}
    state = make_state(active_query="fire pokemon", retrieved_docs=[card])

    with patch("app.agent.nodes.grade.get_client", return_value=mock_client):
        from app.agent.nodes.grade import grade_documents
        result = await grade_documents(state)

    assert len(result["relevant_docs"]) == 1
    assert len(result["graded_docs"]) == 1
    assert result["graded_docs"][0]["grade"] == "relevant"
    assert result["trajectory"][0]["metadata"]["relevant_count"] == 1


@pytest.mark.asyncio
async def test_grade_documents_not_relevant():
    grade_response = '{"grade": "not_relevant", "reasoning": "Wrong type, no matching ability"}'

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text=grade_response)]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    card = {"card_id": "test-2", "name": "Squirtle", "types": ["Water"], "hp": 60}
    state = make_state(active_query="fire pokemon", retrieved_docs=[card])

    with patch("app.agent.nodes.grade.get_client", return_value=mock_client):
        from app.agent.nodes.grade import grade_documents
        result = await grade_documents(state)

    assert len(result["relevant_docs"]) == 0
    assert result["graded_docs"][0]["grade"] == "not_relevant"


# ── Full graph with all LLM/ES mocked ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_graph_simple_path_trajectory_accumulates(monkeypatch):
    """Verify a simple query accumulates one trajectory step per node."""
    from unittest.mock import AsyncMock, patch, MagicMock

    # Lower threshold so 1 relevant card is enough to skip the rewrite loop
    monkeypatch.setenv("MIN_RELEVANT_DOCS", "1")

    # Mocks
    analyze_resp = '{"is_complex": false, "sub_queries": ["charizard"]}'
    grade_resp = '{"grade": "relevant", "reasoning": "Exact name match"}'
    generate_resp = "Charizard is a powerful Fire-type Pokemon."

    mock_card = {"card_id": "xy1-11", "name": "Charizard", "types": ["Fire"], "hp": 250, "stage": "Stage 2"}

    def make_mock_message(text):
        msg = AsyncMock()
        msg.content = [AsyncMock(text=text)]
        return msg

    mock_anthropic = AsyncMock()
    mock_anthropic.messages.create = AsyncMock(side_effect=[
        make_mock_message(analyze_resp),   # analyze_query
        make_mock_message(grade_resp),     # grade_documents (1 card)
        make_mock_message(generate_resp),  # generate_answer
    ])

    with (
        patch("app.agent.nodes.analyze.get_client", return_value=mock_anthropic),
        patch("app.agent.nodes.grade.get_client", return_value=mock_anthropic),
        patch("app.agent.nodes.generate.get_client", return_value=mock_anthropic),
        patch("app.agent.nodes.retrieve.search_for_query", AsyncMock(return_value=[mock_card])),
    ):
        from app.agent.graph import build_graph
        graph = build_graph()
        initial = {
            "original_query": "charizard",
            "is_complex": False,
            "sub_queries": [],
            "active_query": "charizard",
            "retrieved_docs": [],
            "iteration_count": 0,
            "graded_docs": [],
            "relevant_docs": [],
            "final_answer": "",
            "final_cards": [],
            "trajectory": [],
        }
        result = await graph.ainvoke(initial)

    node_names = [s["node"] for s in result["trajectory"]]
    assert "analyze_query" in node_names
    assert "retrieve" in node_names
    assert "grade_documents" in node_names
    assert "generate_answer" in node_names
    assert "rewrite_query" not in node_names  # no loop for simple query
    assert result["final_answer"] == generate_resp
