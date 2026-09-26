"""Restaurant Order Management AI Agent System using LangGraph.

Uses Groq LLM (openai/gpt-oss-120b) for natural-language order extraction and
conversational responses. Implements a full order lifecycle:
receive_order → order_confirm → cook → serve with retry logic at each stage.

Menu:
    burger: 10
    pizza:  20
    momos:   4
"""

from __future__ import annotations

import os
import random
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Menu – dish name → available quantity
# ---------------------------------------------------------------------------
MENU: dict[str, int] = {
    "burger": 10,
    "pizza": 20,
    "momos": 4,
}

# ---------------------------------------------------------------------------
# Groq LLM instance (reused across nodes)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# Pydantic model for structured extraction
# ---------------------------------------------------------------------------


class ParsedOrder(BaseModel):
    """Structured output from the LLM for order extraction."""
    is_food_order: bool = Field(
        description="True if the user message is related to food ordering, False otherwise."
    )
    dish_name: str = Field(
        default="",
        description="The name of the dish the user wants to order (lowercased).",
    )
    quantity: int = Field(
        default=0,
        ge=0,
        description="The number of servings requested. Default to 1 if not specified.",
    )


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class OrderState(TypedDict, total=False):
    """State shared across all nodes in the graph."""
    # Annotated message list (LLM ↔ user conversation history)
    messages: Annotated[list[BaseMessage], add_messages]

    # Order details
    dish_name: str
    required_quantity: int
    available_quantity: int  # Written by order_confirm from the menu

    # Status tracking – updated by each node
    status: str

    # Retry counters (decremented on each failure)
    order_retry_attempts: int   # starts at 3
    cook_retry_attempts: int    # starts at 2
    serve_retry_attempts: int   # starts at 2

    # Final outcome
    final_result: str  # "completed" or "not_completed"


# ---------------------------------------------------------------------------
# Helper: extract order via LLM with structured output
# ---------------------------------------------------------------------------


def _extract_order(user_text: str) -> ParsedOrder:
    """Use the Groq LLM to extract dish name and quantity from user input."""
    structured_llm = llm.with_structured_output(ParsedOrder)
    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a restaurant order extraction assistant. "
                    "Extract the dish name and quantity from the user's message. "
                    "If the user does not specify a quantity, assume 1. "
                    "Normalize the dish name to lowercase. "
                    "If the message is NOT related to food ordering at all "
                    "(e.g. general questions, math, coding, etc.), set is_food_order to false."
                )
            ),
            HumanMessage(content=user_text),
        ]
    )
    return result


# ---------------------------------------------------------------------------
# Node: receive_order
# ---------------------------------------------------------------------------


def receive_order(state: OrderState) -> dict:
    """First node – takes user input, uses LLM to extract order details.

    If the input is not food-related, the LLM rejects it with a polite message.
    """
    user_text = state["messages"][-1].content

    try:
        parsed = _extract_order(str(user_text))
    except Exception as exc:
        return {
            "status": "error",
            "final_result": "not_completed",
            "messages": [
                AIMessage(
                    content=(
                        f"I couldn't process your request due to an error: {exc}. "
                        "Please try again."
                    )
                )
            ],
        }

    # Not a food order
    if not parsed.is_food_order or not parsed.dish_name or parsed.quantity <= 0:
        return {
            "status": "irrelevant",
            "final_result": "not_completed",
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry, I'm an AI agent designed specifically for food ordering "
                        "at this restaurant. I cannot help with general-purpose queries. "
                        "Please place a food order!"
                    )
                )
            ],
        }

    # Valid food order
    return {
        "dish_name": parsed.dish_name,
        "required_quantity": parsed.quantity,
        "status": "received",
        "messages": [
            AIMessage(
                content=(
                    f"Got it! You'd like to order {parsed.quantity} {parsed.dish_name}(s). "
                    "Let me check availability..."
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: order_confirm
# ---------------------------------------------------------------------------


def order_confirm(state: OrderState) -> dict:
    """Checks the menu for dish availability and quantity.

    Sets status to one of:
      - "confirm"        → dish available in full quantity
      - "partial"        → dish available but quantity insufficient
      - "not_available"  → dish not on menu or 0 quantity
    Also writes available_quantity into state.
    """
    dish = state["dish_name"]
    needed = state["required_quantity"]
    available = MENU.get(dish, 0)

    if available == 0:
        status = "not_available"
        msg = (
            f"Sorry, '{dish}' is not available on our menu. "
            "Our menu has: " + ", ".join(f"{d} ({q} available)" for d, q in MENU.items()) + ". "
            "Would you like to order something else?"
        )
    elif available < needed:
        status = "partial"
        msg = (
            f"We only have {available} {dish}(s) available, "
            f"but you requested {needed}. "
            f"Would you like to go ahead with {available}, or place a different order?"
        )
    else:
        status = "confirm"
        msg = (
            f"Great news! {dish} is available. "
            f"Proceeding to prepare your {needed} {dish}(s)."
        )

    return {
        "available_quantity": available,
        "status": status,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Node: handle_user_decision (for partial / not_available)
# ---------------------------------------------------------------------------


def handle_user_decision(state: OrderState) -> dict:
    """Prompts the user when the order is partial or not available.

    The user can:
      - Accept a partial order (type 'yes' or 'partial')
      - Place a new order (type a new dish)
      - Quit (type 'quit')

    Decrements order_retry_attempts each time this node runs.
    """
    status = state["status"]
    attempts_left = state["order_retry_attempts"] - 1

    # If retries exhausted, apologize and end
    if attempts_left <= 0:
        return {
            "order_retry_attempts": 0,
            "status": "order_retries_exhausted",
            "final_result": "not_completed",
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry, we've exceeded the maximum number of order attempts (3). "
                        "Thank you for your patience. Goodbye!"
                    )
                )
            ],
        }

    # Prompt user for a decision
    if status == "not_available":
        prompt = (
            f"'{state['dish_name']}' is not on our menu. "
            "Please enter a new order, or type 'quit' to cancel: "
        )
    else:  # partial
        prompt = (
            f"We have only {state['available_quantity']} {state['dish_name']}(s). "
            f"Type 'yes' to accept {state['available_quantity']}, "
            "enter a new order, or type 'quit' to cancel: "
        )

    answer = input(prompt).strip()

    # User quits
    if answer.lower() == "quit":
        return {
            "order_retry_attempts": attempts_left,
            "status": "cancelled",
            "final_result": "not_completed",
            "messages": [AIMessage(content="Order cancelled. Thank you!")],
        }

    # User accepts partial
    if answer.lower() in ("yes", "partial") and status == "partial":
        available = state["available_quantity"]
        dish = state["dish_name"]
        return {
            "required_quantity": available,
            "order_retry_attempts": attempts_left,
            "status": "confirm",
            "messages": [
                AIMessage(
                    content=f"Confirmed: {available} {dish}(s). Sending to kitchen!"
                )
            ],
        }

    # User places a new order – extract via LLM
    try:
        parsed = _extract_order(answer)
    except Exception:
        return {
            "order_retry_attempts": attempts_left,
            "status": "invalid_retry",
            "messages": [
                AIMessage(content="I couldn't understand that. Please try a food order.")
            ],
        }

    if not parsed.is_food_order or not parsed.dish_name or parsed.quantity <= 0:
        return {
            "order_retry_attempts": attempts_left,
            "status": "invalid_retry",
            "messages": [
                AIMessage(
                    content=(
                        "That doesn't seem like a food order. "
                        "Please enter a valid dish and quantity."
                    )
                )
            ],
        }

    return {
        "dish_name": parsed.dish_name,
        "required_quantity": parsed.quantity,
        "order_retry_attempts": attempts_left,
        "status": "new_order",
        "messages": [
            HumanMessage(content=answer),
            AIMessage(
                content=(
                    f"New order: {parsed.quantity} {parsed.dish_name}(s). "
                    "Checking availability..."
                )
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Node: cook
# ---------------------------------------------------------------------------


def cook(state: OrderState) -> dict:
    """Simulates cooking with 60% success / 40% failure probability.

    cook_retry_attempts tracks how many times cook can be RE-called after
    its initial invocation.  Every call that is NOT the first (status != "confirm")
    consumes one retry.  If retries hit 0 before a call, cook returns failure
    immediately without attempting to cook.
    """
    dish = state["dish_name"]
    qty = state["required_quantity"]
    cook_retries = state["cook_retry_attempts"]

    # Determine if this is the first cook call or a retry
    is_first_call = state.get("status") == "confirm"

    if not is_first_call:
        # This is a retry – consume one retry attempt
        if cook_retries <= 0:
            return {
                "cook_retry_attempts": 0,
                "status": "cook_failed",
                "final_result": "not_completed",
                "messages": [
                    AIMessage(
                        content=(
                            "I'm very sorry, the kitchen has exhausted all retry attempts "
                            f"for your order ({qty} {dish}(s)). "
                            "We sincerely apologize for the inconvenience."
                        )
                    )
                ],
            }
        cook_retries -= 1

    # Attempt cooking (60% success, 40% failure)
    if random.random() < 0.4:
        # Cook failed
        return {
            "cook_retry_attempts": cook_retries,
            "status": "cook_retry",
            "messages": [
                AIMessage(
                    content=(
                        f"Oops! The kitchen hit a snag while preparing your {dish}. "
                        f"Retrying... ({cook_retries} retry(s) remaining)"
                    )
                )
            ],
        }

    # Cook succeeded
    return {
        "cook_retry_attempts": cook_retries,
        "status": "ready",
        "messages": [
            AIMessage(
                content=f"Your {qty} {dish}(s) have been cooked and are ready to serve!"
            )
        ],
    }


# ---------------------------------------------------------------------------
# Node: serve
# ---------------------------------------------------------------------------


def serve(state: OrderState) -> dict:
    """Simulates serving with 60% success / 40% failure probability.

    On failure, decrements serve_retry_attempts.
    If serve fails, it routes back to cook (if cook retries remain).
    If both serve and cook retries exhausted → apology + END.
    On success, sets status to 'complete'.
    """
    dish = state["dish_name"]
    qty = state["required_quantity"]

    if random.random() < 0.4:
        # Serve failed
        left = state["serve_retry_attempts"] - 1
        if left <= 0:
            return {
                "serve_retry_attempts": 0,
                "status": "serve_failed",
                "final_result": "not_completed",
                "messages": [
                    AIMessage(
                        content=(
                            "I'm very sorry, we were unable to serve your order "
                            f"({qty} {dish}(s)) after multiple attempts. "
                            "We sincerely apologize for the inconvenience."
                        )
                    )
                ],
            }
        # Check if cook retries are still available for a re-cook
        cook_left = state.get("cook_retry_attempts", 0)
        if cook_left <= 0:
            return {
                "serve_retry_attempts": left,
                "status": "serve_failed",
                "final_result": "not_completed",
                "messages": [
                    AIMessage(
                        content=(
                            "Serving failed, and the kitchen has no more retry attempts. "
                            "I'm very sorry, we cannot complete your order. "
                            "We sincerely apologize for the inconvenience."
                        )
                    )
                ],
            }
        return {
            "serve_retry_attempts": left,
            "status": "serve_retry",
            "messages": [
                AIMessage(
                    content=(
                        f"Serving your {dish} failed. "
                        f"Sending back to the kitchen to retry... "
                        f"({left} serve attempt(s) remaining)"
                    )
                )
            ],
        }

    # Serve succeeded
    return {
        "status": "complete",
        "final_result": "completed",
        "messages": [
            AIMessage(
                content=(
                    f"🎉 Your order of {qty} {dish}(s) has been served successfully! "
                    "Enjoy your meal!"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------


def route_after_receive(state: OrderState) -> Literal["order_confirm", "end"]:
    """After receive_order: proceed to confirm if valid, else end."""
    if state.get("status") == "received":
        return "order_confirm"
    return "end"


def route_after_confirm(state: OrderState) -> Literal["cook", "handle_user_decision"]:
    """After order_confirm: cook if fully available, else ask user."""
    if state["status"] == "confirm":
        return "cook"
    return "handle_user_decision"


def route_after_user_decision(
    state: OrderState,
) -> Literal["cook", "order_confirm", "end"]:
    """After handle_user_decision: route based on user's choice."""
    status = state["status"]
    if status == "confirm":
        # User accepted partial order
        return "cook"
    if status in ("new_order", "invalid_retry"):
        # User placed a new order or invalid input → re-check
        return "order_confirm"
    # cancelled or order_retries_exhausted
    return "end"


def route_after_cook(state: OrderState) -> Literal["cook", "serve", "end"]:
    """After cook: retry cook, proceed to serve, or end on failure."""
    status = state["status"]
    if status == "cook_retry":
        return "cook"
    if status == "ready":
        return "serve"
    # cook_failed
    return "end"


def route_after_serve(state: OrderState) -> Literal["cook", "end"]:
    """After serve: retry via cook if serve failed but retries remain, else end."""
    status = state["status"]
    if status == "serve_retry":
        return "cook"
    # complete or serve_failed
    return "end"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def build_graph():
    """Construct and compile the order management state graph."""
    graph = StateGraph(OrderState)

    # Add nodes
    graph.add_node("receive_order", receive_order)
    graph.add_node("order_confirm", order_confirm)
    graph.add_node("handle_user_decision", handle_user_decision)
    graph.add_node("cook", cook)
    graph.add_node("serve", serve)

    # Edges
    graph.add_edge(START, "receive_order")

    graph.add_conditional_edges(
        "receive_order",
        route_after_receive,
        {"order_confirm": "order_confirm", "end": END},
    )

    graph.add_conditional_edges(
        "order_confirm",
        route_after_confirm,
        {"cook": "cook", "handle_user_decision": "handle_user_decision"},
    )

    graph.add_conditional_edges(
        "handle_user_decision",
        route_after_user_decision,
        {"cook": "cook", "order_confirm": "order_confirm", "end": END},
    )

    graph.add_conditional_edges(
        "cook",
        route_after_cook,
        {"cook": "cook", "serve": "serve", "end": END},
    )

    graph.add_conditional_edges(
        "serve",
        route_after_serve,
        {"cook": "cook", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the interactive restaurant ordering agent."""
    app = build_graph()

    print("=" * 60)
    print("  🍽️  Welcome to the AI Restaurant Order Agent!")
    print("  Menu:")
    for dish, qty in MENU.items():
        print(f"    • {dish.capitalize():10s} — {qty} available")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    while True:
        order = input("\nWhat would you like to order? ").strip()
        if order.lower() in {"exit", "quit"}:
            print("Thank you for visiting! Goodbye. 👋")
            break

        # Initialize state with the user's order
        initial_state: OrderState = {
            "messages": [HumanMessage(content=order)],
            "order_retry_attempts": 3,
            "cook_retry_attempts": 2,
            "serve_retry_attempts": 2,
            "final_result": "not_completed",
        }

        result = app.invoke(initial_state, {"recursion_limit": 50})

        # Print all AI messages from this run
        print("\n--- Order Transcript ---")
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                print(f"🤖 {msg.content}")
            elif isinstance(msg, HumanMessage):
                print(f"👤 {msg.content}")
        print(f"\n📋 Final Result: {result.get('final_result', 'not_completed')} "
              f"(status: {result.get('status')})")
        print("-" * 40)


if __name__ == "__main__":
    main()
