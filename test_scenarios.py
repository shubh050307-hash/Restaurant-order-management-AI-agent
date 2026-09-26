"""Test scenarios for the restaurant order management AI agent.

Mocks:
  - _extract_order  → avoids real LLM calls
  - builtins.input  → provides scripted user responses
  - random.random   → controls cook / serve outcomes

Run:  .venv/bin/python test_scenarios.py
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from main import OrderState, ParsedOrder, build_graph

# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _run(
    name: str,
    initial_input: str,
    extract_responses: list[ParsedOrder],
    input_responses: list[str],
    random_values: list[float],
    expected_final: str,
) -> bool:
    """Invoke the graph once with controlled mocks and check the outcome."""
    extract_iter = iter(extract_responses)
    input_iter = iter(input_responses)
    random_iter = iter(random_values)

    with (
        patch("main._extract_order", side_effect=lambda _: next(extract_iter)),
        patch("builtins.input", side_effect=lambda _: next(input_iter)),
        patch("main.random.random", side_effect=lambda: next(random_iter)),
    ):
        app = build_graph()
        state: OrderState = {
            "messages": [HumanMessage(content=initial_input)],
            "order_retry_attempts": 3,
            "cook_retry_attempts": 2,
            "serve_retry_attempts": 2,
            "final_result": "not_completed",
        }
        result = app.invoke(state, {"recursion_limit": 50})

    actual = result.get("final_result", "not_completed")
    status = result.get("status", "?")
    passed = actual == expected_final

    print(f"\n{'=' * 65}")
    print(f"  {'✅' if passed else '❌'}  {name}")
    print(f"{'=' * 65}")
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            print(f"  🤖  {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  👤  {msg.content}")
    print(f"\n  final_result = {actual}  |  status = {status}")
    if not passed:
        print(f"  ⚠️  EXPECTED final_result = {expected_final}")
    print("-" * 65)
    return passed


# ──────────────────────────────────────────────────────────────
# TC 1  –  END due to order retry
# ──────────────────────────────────────────────────────────────
#
# Flow (two separate graph invocations):
#
#  Run A: user asks unrelated question → irrelevant → END
#
#  Run B: user orders 10 momos (only 4 available → partial)
#    handle_user_decision #1 (retries 3→2): rejects partial, orders "sushi" (not on menu)
#    order_confirm: not_available
#    handle_user_decision #2 (retries 2→1): orders "biryani" (not on menu)
#    order_confirm: not_available
#    handle_user_decision #3 (retries 1→0): exhausted → END
#
# ──────────────────────────────────────────────────────────────

def tc1() -> bool:
    print("\n\n" + "█" * 65)
    print("  TC1 — Order retry exhaustion")
    print("█" * 65)

    # --- Run A: unrelated question ---
    a = _run(
        "TC1-A  Unrelated question → rejected",
        initial_input="What is the capital of France?",
        extract_responses=[
            ParsedOrder(is_food_order=False, dish_name="", quantity=0),
        ],
        input_responses=[],
        random_values=[],
        expected_final="not_completed",
    )

    # --- Run B: partial → unavailable → retry exhausted ---
    b = _run(
        "TC1-B  Partial → reject → unavailable → retry exhausted",
        initial_input="I want 10 momos",
        extract_responses=[
            # receive_order extracts the initial order
            ParsedOrder(is_food_order=True, dish_name="momos", quantity=10),
            # handle_user_decision #1: user types "sushi"
            ParsedOrder(is_food_order=True, dish_name="sushi", quantity=1),
            # handle_user_decision #2: user types "biryani"
            ParsedOrder(is_food_order=True, dish_name="biryani", quantity=1),
        ],
        input_responses=[
            "I want sushi",     # reject partial, order sushi
            "I want biryani",   # sushi unavailable, try biryani
        ],
        random_values=[],       # no cook/serve involved
        expected_final="not_completed",
    )

    return a and b


# ──────────────────────────────────────────────────────────────
# TC 2  –  Overall SUCCESS
# ──────────────────────────────────────────────────────────────
#
# Flow:
#   order "5 burgers" → available (10 in stock)
#   cook #1 (first call, retries=2): FAIL       → cook_retry
#   cook #2 (retry, retries 2→1):    SUCCESS    → serve
#   serve #1 (retries=2):            FAIL       → serve_retry → cook
#   cook #3 (retry, retries 1→0):    SUCCESS    → serve
#   serve #2 (retries=1):            SUCCESS    → complete ✅
#
# random sequence: 0.1(cook fail), 0.5(cook ok), 0.1(serve fail), 0.5(cook ok), 0.5(serve ok)
#
# ──────────────────────────────────────────────────────────────

def tc2() -> bool:
    print("\n\n" + "█" * 65)
    print("  TC2 — Overall success with cook & serve retries")
    print("█" * 65)

    return _run(
        "TC2  cook-fail → cook-ok → serve-fail → cook-ok → serve-ok",
        initial_input="I want 5 burgers",
        extract_responses=[
            ParsedOrder(is_food_order=True, dish_name="burger", quantity=5),
        ],
        input_responses=[],
        random_values=[0.1, 0.5, 0.1, 0.5, 0.5],
        expected_final="completed",
    )


# ──────────────────────────────────────────────────────────────
# TC 3  –  Overall FAIL (cook retries exhausted)
# ──────────────────────────────────────────────────────────────
#
# Flow:
#   order "8 momos" → partial (only 4 available)
#   handle_user_decision (retries 3→2): reject, order "5 pizza"
#   order_confirm: pizza available (20) → confirm → cook
#   cook #1 (first, retries=2):   FAIL      → cook_retry
#   cook #2 (retry, retries 2→1): SUCCESS   → serve
#   serve #1 (retries=2):         FAIL      → serve_retry → cook
#   cook #3 (retry, retries 1→0): SUCCESS   → serve
#   serve #2 (retries=1):         FAIL      → serve_retries=0 → serve_failed → END
#       (cook_retries=0 as well, so cook can't retry either)
#
# random sequence: 0.1(cook fail), 0.5(cook ok), 0.1(serve fail), 0.5(cook ok), 0.1(serve fail)
#
# ──────────────────────────────────────────────────────────────

def tc3() -> bool:
    print("\n\n" + "█" * 65)
    print("  TC3 — Overall FAIL (cook & serve retries exhausted)")
    print("█" * 65)

    return _run(
        "TC3  partial → new order → cook retries & serve retries exhaust",
        initial_input="I want 8 momos",
        extract_responses=[
            # receive_order
            ParsedOrder(is_food_order=True, dish_name="momos", quantity=8),
            # handle_user_decision: user orders pizza
            ParsedOrder(is_food_order=True, dish_name="pizza", quantity=5),
        ],
        input_responses=[
            "I want 5 pizza",   # reject partial momos, order pizza
        ],
        random_values=[0.1, 0.5, 0.1, 0.5, 0.1],
        expected_final="not_completed",
    )


# ──────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [tc1(), tc2(), tc3()]

    print("\n\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    labels = ["TC1", "TC2", "TC3"]
    all_pass = True
    for label, ok in zip(labels, results):
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {label}: {mark}")
        if not ok:
            all_pass = False
    print("=" * 65)

    sys.exit(0 if all_pass else 1)
