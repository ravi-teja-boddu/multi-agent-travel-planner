"""Multi-Agent Travel Planner.

A LangGraph system that routes a travel request to specialized agents
(flight/hotel search and itinerary planning), runs them in parallel, and
merges their answers into one response.

Typical use::

    from travel_planner import TravelPlanner

    planner = TravelPlanner()
    print(planner.ask("Plan a 3-day trip to Rome", thread_id="me"))
"""

from travel_planner.graph import TravelPlanner, build_travel_planner_graph

__all__ = ["TravelPlanner", "build_travel_planner_graph"]
__version__ = "0.1.0"
