"""Command-line interface for the Multi-Agent Travel Planner.

Examples::

    # Interactive chat (remembers context between questions)
    python main.py

    # One question, then exit
    python main.py "What is the best time to visit Jaipur?"

    # Show routing and each agent's raw answer as well
    python main.py --verbose "Plan 3 days in Rome and find hotels for 2 adults, 10-13 Mar 2027"

    # Print the graph as a Mermaid diagram
    python main.py --graph
"""

from __future__ import annotations

import argparse
import sys

from travel_planner.config import MissingAPIKeyError
from travel_planner.graph import DEFAULT_RECURSION_LIMIT, TravelPlanner

RULE = "=" * 60
EXIT_WORDS = {"exit", "quit", "q", ":q"}


def print_verbose(state: dict) -> None:
    """Print routing decisions and each agent's raw answer."""
    print(RULE)
    print("TASKS DISPATCHED:")
    for task in state.get("tasks", []):
        print(f"  - {task.source}: {task.user_query}")
    print(RULE)
    print("AGENT RESULTS:")
    for result in state.get("agent_results", []):
        print(f"\n[{result['agent'].upper()}]")
        print("-" * 40)
        print(result["result"])
    print(RULE)
    print("FINAL RESPONSE:")
    print(RULE)


def answer(planner: TravelPlanner, query: str, thread_id: str, verbose: bool, recursion_limit: int) -> None:
    if verbose:
        print("\nWorking...", flush=True)
        for node, _ in planner.stream_updates(query, thread_id=thread_id, recursion_limit=recursion_limit):
            print(f"  ✓ {node}", flush=True)
        state = planner.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        print_verbose(state)
    else:
        print("\nThinking... (itinerary requests can take a minute or two)\n", flush=True)
        state = planner.run(query, thread_id=thread_id, recursion_limit=recursion_limit)
    print(state.get("final_answer") or "(no answer)")
    print()


def chat(planner: TravelPlanner, thread_id: str, verbose: bool, recursion_limit: int) -> None:
    print("Multi-Agent Travel Planner — ask about flights, hotels, itineraries or travel tips.")
    print(f"Session: {thread_id}. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            break
        try:
            answer(planner, query, thread_id, verbose, recursion_limit)
        except KeyboardInterrupt:
            print("\n(cancelled)\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Travel Planner (LangGraph).")
    parser.add_argument("query", nargs="*", help="Question to ask. Leave empty for interactive chat.")
    parser.add_argument("-t", "--thread-id", help="Conversation id (default: a new random id).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show routing and each agent's answer.")
    parser.add_argument("--graph", action="store_true", help="Print the graph as Mermaid and exit.")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=DEFAULT_RECURSION_LIMIT,
        help=f"Max graph steps per question (default: {DEFAULT_RECURSION_LIMIT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        planner = TravelPlanner()
    except MissingAPIKeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.graph:
        print(planner.mermaid())
        return 0

    thread_id = args.thread_id or planner.new_thread_id()
    if args.query:
        answer(planner, " ".join(args.query), thread_id, args.verbose, args.recursion_limit)
    else:
        chat(planner, thread_id, args.verbose, args.recursion_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
