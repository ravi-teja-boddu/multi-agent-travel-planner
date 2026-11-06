"""Chat model, the itinerary deep-research agent, and standalone agents.

The main graph (``travel_planner.graph``) uses:

* :func:`create_llm` - the shared OpenAI chat model.
* :func:`create_itinerary_tool` - a tool that wraps the deep-research
  itinerary agent, so the planner can hand off day-by-day planning.

:func:`create_search_agent` and :func:`create_travel_scout` build the same
agents as ready-made LangChain agents. They are handy for testing one agent
on its own, outside the full graph.
"""

from __future__ import annotations

from typing import Optional

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from travel_planner.config import Settings
from travel_planner.prompts import (
    FLIGHT_HOTEL_SEARCH_PROMPT,
    ITINERARY_RESEARCH_PROMPT,
    TRAVEL_SCOUT_PROMPT,
)
from travel_planner.tools import SEARCH_TOOLS


def create_llm(settings: Optional[Settings] = None) -> ChatOpenAI:
    """Create the OpenAI chat model used by every agent."""
    settings = settings or Settings.from_env()
    return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)


def create_itinerary_research_agent(llm: BaseChatModel, internet_search: BaseTool):
    """Create the deep-research agent that writes detailed itineraries.

    It plans its own research steps and calls ``internet_search`` as often
    as it needs before writing the final itinerary.
    """
    return create_deep_agent(
        model=llm,
        system_prompt=ITINERARY_RESEARCH_PROMPT,
        tools=[internet_search],
    )


def create_itinerary_tool(llm: BaseChatModel, internet_search: BaseTool) -> BaseTool:
    """Wrap the itinerary deep-research agent as a tool (a sub-agent)."""
    itinerary_agent = create_itinerary_research_agent(llm, internet_search)

    @tool(
        "itinerary_research_agent",
        description=(
            "Plans a detailed, researched, day-by-day travel itinerary. "
            "Pass a complete request: destination, number of days, season or dates, "
            "traveler interests and budget."
        ),
    )
    def call_itinerary_research_agent(query: str) -> str:
        result = itinerary_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].text

    return call_itinerary_research_agent


def create_search_agent(llm: BaseChatModel):
    """Standalone flight & hotel search agent."""
    return create_agent(model=llm, tools=SEARCH_TOOLS, system_prompt=FLIGHT_HOTEL_SEARCH_PROMPT)


def create_travel_scout(llm: BaseChatModel, internet_search: BaseTool, itinerary_tool: BaseTool):
    """Standalone travel scout: general questions plus itinerary hand-off."""
    return create_agent(
        model=llm,
        tools=[internet_search, itinerary_tool],
        system_prompt=TRAVEL_SCOUT_PROMPT,
    )
