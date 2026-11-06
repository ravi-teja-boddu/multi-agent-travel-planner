"""Offline tests for the travel planner graph.

A scripted fake LLM stands in for OpenAI, and fake tools stand in for
SerpAPI/Tavily, so these tests need no API keys or network.

Run with:  pytest -q
"""

from __future__ import annotations

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from travel_planner.graph import AgentTask, ClassificationResult, build_travel_planner_graph
from travel_planner.prompts import FLIGHT_HOTEL_SEARCH_PROMPT, TRAVEL_SCOUT_PROMPT


@tool
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str) -> str:
    """Fake flight search."""
    return f'[{{"price": 450, "route": "{departure_airport}-{arrival_airport}", "date": "{outbound_date}"}}]'


@tool
def tavily_search(query: str) -> str:
    """Fake web search."""
    return f"results for {query}"


class FakeLLM:
    """Minimal stand-in for a chat model, scripted per agent."""

    def __init__(self, routes):
        self.routes = list(routes)  # one ClassificationResult per turn
        self.router_inputs = []
        self.search_inputs = []
        self.planner_inputs = []
        self.synth_calls = 0

    # The graph calls bind_tools twice and with_structured_output once.
    def bind_tools(self, tools):
        names = {t.name for t in tools}
        return _Bound(self, "search" if "search_flights" in names else "planner")

    def with_structured_output(self, schema):
        return _Router(self)

    def invoke(self, messages):  # synthesizer
        self.synth_calls += 1
        return AIMessage(content="MERGED: " + messages[0].content[-40:])


class _Router:
    def __init__(self, fake):
        self.fake = fake

    def invoke(self, messages):
        self.fake.router_inputs.append(messages[-1].content)
        return self.fake.routes.pop(0)


class _Bound:
    def __init__(self, fake, kind):
        self.fake, self.kind = fake, kind

    def invoke(self, messages):
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage), "sub-query must stay in the agent transcript"
        if self.kind == "search":
            assert messages[0].content == FLIGHT_HOTEL_SEARCH_PROMPT
            self.fake.search_inputs.append(messages)
            if len(messages) == 2:  # first call: ask for a tool
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "search_flights",
                        "args": {"departure_airport": "JFK", "arrival_airport": "LHR", "outbound_date": "2027-01-21"},
                        "id": "call_1",
                    }],
                )
            assert messages[-1].type == "tool"
            return AIMessage(content="Cheapest flight: $450")
        assert messages[0].content == TRAVEL_SCOUT_PROMPT
        self.fake.planner_inputs.append(messages)
        return AIMessage(content=f"Plan for: {messages[1].content}")


def build(fake):
    return build_travel_planner_graph(
        llm=fake,
        search_tools=[search_flights],
        planner_tools=[tavily_search],
        checkpointer=InMemorySaver(),
    )


def run(graph, query, thread="t1"):
    return graph.invoke(
        {"user_query": query, "messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": thread}},
    )


def test_parallel_agents_are_merged_once_both_finish():
    fake = FakeLLM([
        ClassificationResult(tasks=[
            AgentTask(source="search_agent_node", user_query="Flights JFK to LHR", focus="prices"),
            AgentTask(source="planner_agent_node", user_query="5 days in London"),
        ], requires_synthesis=True),
    ])
    state = run(build(fake), "Plan London and find flights")

    assert {r["agent"] for r in state["agent_results"]} == {"search_agent", "planner_agent"}
    # Planner finishes before search; the synthesizer must wait and merge only once.
    assert fake.synth_calls == 1
    assert state["final_answer"].startswith("MERGED")
    assert "Focus on: prices" in fake.search_inputs[0][1].content


def test_follow_up_turn_resets_state_and_sees_history():
    fake = FakeLLM([
        ClassificationResult(tasks=[AgentTask(source="planner_agent_node", user_query="Best time for Jaipur")]),
        ClassificationResult(tasks=[AgentTask(source="planner_agent_node", user_query="Historical sites in Jaipur")]),
    ])
    graph = build(fake)
    first = run(graph, "What is the best time to visit Jaipur?")
    second = run(graph, "Major historical sites there")

    assert first["final_answer"] == "Plan for: Best time for Jaipur"
    assert second["final_answer"] == "Plan for: Historical sites in Jaipur"
    # Per-turn results do not leak into the next turn.
    assert len(second["agent_results"]) == 1
    assert len(second["planner_messages"]) == 2
    # The router saw the earlier turn when routing the follow-up.
    assert "best time to visit Jaipur" in fake.router_inputs[1]
    assert "Major historical sites there" in fake.router_inputs[1]
    # Conversation history kept: 2 user + 2 assistant messages.
    assert len(second["messages"]) == 4
    assert fake.synth_calls == 0


def test_duplicate_and_empty_routes():
    fake = FakeLLM([
        ClassificationResult(tasks=[
            AgentTask(source="planner_agent_node", user_query="a"),
            AgentTask(source="planner_agent_node", user_query="b"),
        ]),
        ClassificationResult(tasks=[]),
    ])
    graph = build(fake)
    assert run(graph, "q1")["final_answer"] == "Plan for: a"
    assert run(graph, "q2")["final_answer"] == "Plan for: q2"  # falls back to planner
