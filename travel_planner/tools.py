"""Tools the agents can call.

* ``search_flights`` / ``search_hotels`` use SerpAPI's Google Flights and
  Google Hotels engines.
* ``create_internet_search`` builds a Tavily web-search tool.

Tools return JSON strings so the LLM can read them directly. Errors are
returned as ``{"error": ...}`` instead of raised, so one failed lookup does
not crash the whole run.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from serpapi import GoogleSearch

MAX_FLIGHT_RESULTS = 10
MAX_HOTEL_RESULTS = 5


# ---------------------------------------------------------------------------
# Flight search
# ---------------------------------------------------------------------------
class FlightSearchInput(BaseModel):
    """Input schema for flight search."""

    departure_airport: str = Field(..., description="Departure airport code (e.g., 'JFK')")
    arrival_airport: str = Field(..., description="Arrival airport code (e.g., 'LAX')")
    outbound_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD) for round trips")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    stops: Optional[int] = Field(None, description="0=Any, 1=Nonstop, 2=1 stop or fewer, 3=2 stops or fewer")


@tool(args_schema=FlightSearchInput)
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    stops: Optional[int] = None,
) -> str:
    """Search for flights between airports (Google Flights via SerpAPI). Prices are in USD."""
    params = {
        "api_key": os.environ.get("SERPAPI_API_KEY"),
        "engine": "google_flights",
        "departure_id": departure_airport.upper(),
        "arrival_id": arrival_airport.upper(),
        "outbound_date": outbound_date,
        "adults": adults,
        "children": children,
        "currency": "USD",
        "type": 1 if return_date else 2,  # 1 = round trip, 2 = one way
    }
    if return_date:
        params["return_date"] = return_date
    if stops is not None:
        params["stops"] = stops

    try:
        results = GoogleSearch(params).get_dict()
        if "error" in results:
            return json.dumps({"error": results["error"]})

        flights = results.get("best_flights", []) + results.get("other_flights", [])
        if not flights:
            return json.dumps({"message": "No flights found"})

        output = []
        for flight in flights[:MAX_FLIGHT_RESULTS]:
            output.append(
                {
                    "price": flight.get("price"),
                    "duration_mins": flight.get("total_duration"),
                    "layovers": [
                        {"airport": lay.get("id"), "duration_mins": lay.get("duration")}
                        for lay in flight.get("layovers", [])
                    ],
                    "airline_logo": flight.get("airline_logo"),
                    "legs": [
                        {
                            "airline": leg.get("airline"),
                            "flight_number": leg.get("flight_number"),
                            "travel_class": leg.get("travel_class"),
                            "departure": f"{leg.get('departure_airport', {}).get('id')} "
                            f"{leg.get('departure_airport', {}).get('time')}",
                            "arrival": f"{leg.get('arrival_airport', {}).get('id')} "
                            f"{leg.get('arrival_airport', {}).get('time')}",
                            "airline_logo": leg.get("airline_logo"),
                        }
                        for leg in flight.get("flights", [])
                    ],
                }
            )
        return json.dumps(output, indent=2)

    except Exception as exc:  # network / API errors are reported to the LLM
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Hotel search
# ---------------------------------------------------------------------------
class HotelSearchInput(BaseModel):
    """Input schema for hotel search."""

    location: str = Field(..., description="Location (e.g., 'New York', 'Paris')")
    check_in_date: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(..., description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    rooms: int = Field(1, description="Number of rooms")
    hotel_class: Optional[str] = Field(None, description="Star rating filter (e.g., '3,4,5')")
    sort_by: int = Field(8, description="3=Lowest price, 8=Highest rating, 13=Most reviewed")


@tool(args_schema=HotelSearchInput)
def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    hotel_class: Optional[str] = None,
    sort_by: int = 8,
) -> str:
    """Search for hotels in a location (Google Hotels via SerpAPI). Prices are in USD."""
    params = {
        "api_key": os.environ.get("SERPAPI_API_KEY"),
        "engine": "google_hotels",
        "q": location,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "rooms": rooms,
        "sort_by": sort_by,
        "currency": "USD",
        "hl": "en",
        "gl": "us",
    }
    if hotel_class:
        params["hotel_class"] = hotel_class

    try:
        results = GoogleSearch(params).get_dict()
        if "error" in results:
            return json.dumps({"error": results["error"]})

        properties = results.get("properties", [])
        if not properties:
            return json.dumps({"message": "No hotels found"})

        output = []
        for hotel in properties[:MAX_HOTEL_RESULTS]:
            images = hotel.get("images") or [{}]
            output.append(
                {
                    "name": hotel.get("name"),
                    "price_per_night": (hotel.get("rate_per_night") or {}).get("lowest"),
                    "total_price": (hotel.get("total_rate") or {}).get("lowest"),
                    "rating": hotel.get("overall_rating"),
                    "reviews": hotel.get("reviews"),
                    "stars": hotel.get("hotel_class"),
                    "amenities": hotel.get("amenities", [])[:8],
                    "thumbnail": images[0].get("thumbnail"),
                }
            )
        return json.dumps(output, indent=2)

    except Exception as exc:  # network / API errors are reported to the LLM
        return json.dumps({"error": str(exc)})


SEARCH_TOOLS = [search_flights, search_hotels]


# ---------------------------------------------------------------------------
# Internet search
# ---------------------------------------------------------------------------
def create_internet_search(max_results: int = 5) -> TavilySearch:
    """Create the Tavily web-search tool. Needs ``TAVILY_API_KEY`` to be set."""
    return TavilySearch(
        max_results=max_results,
        topic="general",
        include_images=True,
        include_image_descriptions=True,
        search_depth="advanced",
    )
