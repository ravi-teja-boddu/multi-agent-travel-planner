"""The multi-agent LangGraph workflow.

Flow for one user turn::

    START -> orchestrator --(Send, in parallel)--> search_agent_node  <-> search_tool_node
                                              \\-> planner_agent_node <-> planner_tool_node
          both agents -> synthesizer -> END

* **orchestrator** - classifies the request and splits it into one task
  per agent, each with a self-contained sub-query.
* **search_agent_node** - flight & hotel search (SerpAPI tools).
* **planner_agent_node** - travel scout: web search for general questions,
  or the itinerary deep-research sub-agent for day-by-day plans.
* **synthesizer** - waits for every dispatched agent, then merges their
  answers (or passes a single answer through unchanged).

With a checkpointer, ``messages`` keeps the conversation across turns, so
follow-up questions ("major sites there?") are understood. Per-turn fields
are cleared by the orchestrator at the start of every turn.
"""

from __future__ import annotations

import uuid
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from travel_planner.prompts import (
    FLIGHT_HOTEL_SEARCH_PROMPT,
    ROUTER_PROMPT,
    SYNTHESIZER_PROMPT,
    TRAVEL_SCOUT_PROMPT,
)

SEARCH_AGENT = "search_agent_node"
PLANNER_AGENT = "planner_agent_node"
SEARCH_TOOLS_NODE = "search_tool_node"
PLANNER_TOOLS_NODE = "planner_tool_node"
ORCHESTRATOR = "orchestrator"
SYNTHESIZER = "synthesizer"

# How many earlier messages the router sees when resolving follow-ups.
ROUTER_HISTORY_LIMIT = 10
DEFAULT_RECURSION_LIMIT = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def extend_or_reset(existing: Optional[list], update: Optional[list]) -> list:
    """Reducer: append ``update`` to ``existing``; an update of ``None`` clears the list.

    Parallel agents can append to the same list safely, and the orchestrator
    can still start each turn with empty per-turn lists.
    """
    if update is None:
        return []
    return (existing or []) + list(update)


def add_messages_list(existing: Optional[list], update: Optional[list]) -> list:
    """Reducer for the long-lived conversation history (append only)."""
    return (existing or []) + list(update or [])


class AgentTask(BaseModel):
    """One agent task with a targeted query."""

    source: Literal["search_agent_node", "planner_agent_node"]
    user_query: str
    focus: str = Field(default="", description="What aspect this agent should focus on")


class ClassificationResult(BaseModel):
    """Router output: one or more tasks to run in parallel."""

    tasks: List[AgentTask] = Field(description="List of agents to invoke with their targeted queries")
    requires_synthesis: bool = Field(
        default=False,
        description="Whether multiple agents are being used and synthesis is needed",
    )


class TravelPlannerState(TypedDict, total=False):
    """Graph state."""

    # Conversation history (kept across turns when a checkpointer is used).
    messages: Annotated[List[AnyMessage], add_messages_list]
    # The latest user request.
    user_query: str
    # Tasks chosen by the router for this turn.
    tasks: List[AgentTask]
    requires_synthesis: bool
    # Answers from the agents this turn. Parallel agents append here.
    agent_results: Annotated[List[dict], extend_or_reset]
    # Merged answer for this turn.
    final_answer: str
    # Private tool-calling transcripts, one per agent, cleared every turn.
    search_messages: Annotated[List[AnyMessage], extend_or_reset]
    planner_messages: Annotated[List[AnyMessage], extend_or_reset]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_tool_calls(message: AnyMessage, tools_by_name: dict[str, BaseTool]) -> list[ToolMessage]:
    """Run every tool call on ``message`` and return one ToolMessage per call.

    Every call gets a reply (even unknown tools or failures), because the
    OpenAI API rejects a conversation with unanswered tool calls.
    """
    results: list[ToolMessage] = []
    for call in getattr(message, "tool_calls", None) or []:
        selected = tools_by_name.get(call["name"])
        if selected is None:
            results.append(
                ToolMessage(content=f"Error: unknown tool '{call['name']}'.", tool_call_id=call["id"])
            )
            continue
        try:
            # Invoking with the full tool call returns a ready-made ToolMessage.
            results.append(selected.invoke(call))
        except Exception as exc:
            results.append(ToolMessage(content=f"Error running {call['name']}: {exc}", tool_call_id=call["id"]))
    return results


def _has_tool_calls(messages: Optional[list]) -> bool:
    return bool(messages) and bool(getattr(messages[-1], "tool_calls", None))


def _format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_travel_planner_graph(
    llm: BaseChatModel,
    search_tools: list[BaseTool],
    planner_tools: list[BaseTool],
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Build and compile the travel planner graph.

    Args:
        llm: Chat model shared by the router, both agents and the synthesizer.
        search_tools: Tools for the search agent (flights, hotels).
        planner_tools: Tools for the planner agent (web search, itinerary sub-agent).
        checkpointer: Optional LangGraph checkpointer for multi-turn memory.
    """
    search_tools_by_name = {t.name: t for t in search_tools}
    planner_tools_by_name = {t.name: t for t in planner_tools}
    search_model = llm.bind_tools(search_tools)
    planner_model = llm.bind_tools(planner_tools)
    router_model = llm.with_structured_output(ClassificationResult)

    # -- Orchestrator -------------------------------------------------------
    def orchestrator(state: TravelPlannerState) -> dict:
        """Split the request into per-agent tasks and clear per-turn state."""
        query = state["user_query"]
        history = [
            m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
        ]
        # The latest message is the current query; everything before is context.
        if history and isinstance(history[-1], HumanMessage) and history[-1].text == query:
            history = history[:-1]
        history = history[-ROUTER_HISTORY_LIMIT:]

        user_content = query
        if history:
            user_content = (
                f"Earlier conversation:\n{_format_history(history)}\n\n"
                f"Latest request (route this one):\n{query}"
            )

        result: ClassificationResult = router_model.invoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user_content)]
        )

        # Run each agent at most once; fall back to the planner if nothing was chosen.
        tasks: list[AgentTask] = []
        for task in result.tasks:
            if task.source not in {t.source for t in tasks}:
                tasks.append(task)
        if not tasks:
            tasks = [AgentTask(source=PLANNER_AGENT, user_query=query)]

        return {
            "tasks": tasks,
            "requires_synthesis": len(tasks) > 1,
            "final_answer": "",
            # None tells the reducer to clear these lists for the new turn.
            "agent_results": None,
            "search_messages": None,
            "planner_messages": None,
        }

    def dispatch_to_agents(state: TravelPlannerState) -> list[Send]:
        """Start every task in parallel with LangGraph's Send API."""
        sends = []
        for task in state.get("tasks", []):
            sub_query = task.user_query
            if task.focus:
                sub_query += f"\n\nFocus on: {task.focus}"
            sends.append(Send(task.source, {"user_query": sub_query}))
        return sends

    # -- Search agent -------------------------------------------------------
    def search_agent_node(state: TravelPlannerState) -> dict:
        """Flight & hotel agent. Loops with search_tool_node until it stops calling tools."""
        history = list(state.get("search_messages") or [])
        new_messages: list[AnyMessage] = []
        if not history:  # first call this turn: start the transcript with the sub-query
            new_messages.append(HumanMessage(content=state.get("user_query", "")))

        response = search_model.invoke(
            [SystemMessage(content=FLIGHT_HOTEL_SEARCH_PROMPT)] + history + new_messages
        )
        new_messages.append(response)

        update: dict = {"search_messages": new_messages}
        if not response.tool_calls:
            update["agent_results"] = [
                {"agent": "search_agent", "focus": "flights and hotels", "result": response.text}
            ]
        return update

    def search_tool_node(state: TravelPlannerState) -> dict:
        msgs = state.get("search_messages") or []
        return {"search_messages": _run_tool_calls(msgs[-1], search_tools_by_name) if msgs else []}

    def route_search(state: TravelPlannerState) -> str:
        return SEARCH_TOOLS_NODE if _has_tool_calls(state.get("search_messages")) else SYNTHESIZER

    # -- Planner agent ------------------------------------------------------
    def planner_agent_node(state: TravelPlannerState) -> dict:
        """Travel scout agent. Loops with planner_tool_node until it stops calling tools."""
        history = list(state.get("planner_messages") or [])
        new_messages: list[AnyMessage] = []
        if not history:
            new_messages.append(HumanMessage(content=state.get("user_query", "")))

        response = planner_model.invoke(
            [SystemMessage(content=TRAVEL_SCOUT_PROMPT)] + history + new_messages
        )
        new_messages.append(response)

        update: dict = {"planner_messages": new_messages}
        if not response.tool_calls:
            update["agent_results"] = [
                {"agent": "planner_agent", "focus": "travel planning and itinerary", "result": response.text}
            ]
        return update

    def planner_tool_node(state: TravelPlannerState) -> dict:
        msgs = state.get("planner_messages") or []
        return {"planner_messages": _run_tool_calls(msgs[-1], planner_tools_by_name) if msgs else []}

    def route_planner(state: TravelPlannerState) -> str:
        return PLANNER_TOOLS_NODE if _has_tool_calls(state.get("planner_messages")) else SYNTHESIZER

    # -- Synthesizer --------------------------------------------------------
    def synthesizer(state: TravelPlannerState) -> dict:
        """Merge agent answers once every dispatched agent has finished."""
        results = state.get("agent_results") or []
        expected = len(state.get("tasks") or [])

        # Agents can finish at different steps; wait for the slower one.
        if len(results) < expected:
            return {}

        if not results:
            answer = "I couldn't process your request."
        elif len(results) == 1:
            answer = results[0]["result"]
        else:
            formatted = ("\n\n" + "=" * 50 + "\n\n").join(
                f"**{r['agent'].upper()}**\nFocus: {r.get('focus', 'N/A')}\n\n{r['result']}"
                for r in results
            )
            response = llm.invoke(
                [HumanMessage(content=SYNTHESIZER_PROMPT.format(query=state["user_query"], results=formatted))]
            )
            answer = response.text

        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    # -- Wiring -------------------------------------------------------------
    builder = StateGraph(TravelPlannerState)
    builder.add_node(ORCHESTRATOR, orchestrator)
    builder.add_node(SEARCH_AGENT, search_agent_node)
    builder.add_node(SEARCH_TOOLS_NODE, search_tool_node)
    builder.add_node(PLANNER_AGENT, planner_agent_node)
    builder.add_node(PLANNER_TOOLS_NODE, planner_tool_node)
    builder.add_node(SYNTHESIZER, synthesizer)

    builder.add_edge(START, ORCHESTRATOR)
    builder.add_conditional_edges(ORCHESTRATOR, dispatch_to_agents, [SEARCH_AGENT, PLANNER_AGENT])

    builder.add_conditional_edges(SEARCH_AGENT, route_search, [SEARCH_TOOLS_NODE, SYNTHESIZER])
    builder.add_edge(SEARCH_TOOLS_NODE, SEARCH_AGENT)

    builder.add_conditional_edges(PLANNER_AGENT, route_planner, [PLANNER_TOOLS_NODE, SYNTHESIZER])
    builder.add_edge(PLANNER_TOOLS_NODE, PLANNER_AGENT)

    builder.add_edge(SYNTHESIZER, END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
class TravelPlanner:
    """High-level entry point: builds the real agents and keeps chat memory.

    Example::

        planner = TravelPlanner()
        planner.ask("What is the best time to visit Jaipur?", thread_id="user-01")
        planner.ask("Major historical sites there?", thread_id="user-01")
    """

    def __init__(self, settings=None, checkpointer: Optional[BaseCheckpointSaver] = None):
        # Imported here so the graph module can be used (and tested) without API keys.
        from travel_planner.agents import create_itinerary_tool, create_llm
        from travel_planner.config import Settings
        from travel_planner.tools import SEARCH_TOOLS, create_internet_search

        self.settings = settings or Settings.from_env()
        self.settings.require_keys()

        llm = create_llm(self.settings)
        internet_search = create_internet_search()
        itinerary_tool = create_itinerary_tool(llm, internet_search)

        self.graph = build_travel_planner_graph(
            llm=llm,
            search_tools=SEARCH_TOOLS,
            planner_tools=[internet_search, itinerary_tool],
            checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        )

    @staticmethod
    def new_thread_id() -> str:
        return f"thread-{uuid.uuid4().hex[:8]}"

    def run(self, query: str, thread_id: Optional[str] = None, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> dict:
        """Run one turn and return the full final state."""
        config = {
            "configurable": {"thread_id": thread_id or self.new_thread_id()},
            "recursion_limit": recursion_limit,
        }
        return self.graph.invoke(
            {"user_query": query, "messages": [HumanMessage(content=query)]},
            config=config,
        )

    def ask(self, query: str, thread_id: Optional[str] = None) -> str:
        """Run one turn and return only the final answer."""
        return self.run(query, thread_id=thread_id)["final_answer"]

    def stream_updates(self, query: str, thread_id: Optional[str] = None, recursion_limit: int = DEFAULT_RECURSION_LIMIT):
        """Yield ``(node_name, update)`` pairs as each node finishes."""
        config = {
            "configurable": {"thread_id": thread_id or self.new_thread_id()},
            "recursion_limit": recursion_limit,
        }
        for chunk in self.graph.stream(
            {"user_query": query, "messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                yield node, update

    def mermaid(self) -> str:
        """Return the graph as a Mermaid diagram."""
        return self.graph.get_graph().draw_mermaid()
