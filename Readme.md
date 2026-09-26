# 🍽️ Restaurant Order Management AI Agent

An intelligent restaurant order management system built with **LangGraph** and **Groq LLM**. The agent handles the complete order lifecycle — from taking orders via natural language to cooking and serving — with built-in retry logic and graceful error handling.

## Architecture

```
User Input → receive_order → order_confirm → cook → serve → END (✅ completed)
                  │                │            │       │
                  │ (irrelevant)   │ (partial/  │(fail) │ (fail → re-cook)
                  └→ END           │ unavail.)  └→cook  └→ cook
                                   ↓                         │
                          handle_user_decision          (if retries
                            │         │                  exhausted)
                            │         └→ END                 │
                            └→ order_confirm              └→ END (❌ fail)
```

### Nodes

| Node | Description |
|------|-------------|
| **receive_order** | Takes user input, uses LLM to extract dish name & quantity. Rejects non-food queries. |
| **order_confirm** | Checks the menu for availability. Sets status: `confirm`, `partial`, or `not_available`. |
| **handle_user_decision** | Prompts user when order is partial/unavailable. User can accept partial, place new order, or quit. |
| **cook** | Simulates cooking (60% success / 40% failure). |
| **serve** | Simulates serving (60% success / 40% failure). Routes back to cook on failure. |

### Retry Logic

| Stage | Max Retries | Behavior on Exhaustion |
|-------|-------------|----------------------|
| **Order** | 3 attempts | Apology → END |
| **Cook** | 2 retries | Apology → END |
| **Serve** | 2 retries | Apology → END (also checks if cook can retry) |

> **Note:** Cook retries are shared — they are consumed whether cook is retried due to its own failure or due to a serve failure routing back to cook.

## Menu

| Dish | Available Quantity |
|------|--------------------|
| Burger | 10 |
| Pizza | 20 |
| Momos | 4 |

## State Schema

```python
class OrderState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history
    dish_name: str                  # Extracted dish name
    required_quantity: int          # User's requested quantity
    available_quantity: int         # Quantity available (from menu)
    status: str                    # Current status (received, confirm, partial, etc.)
    order_retry_attempts: int      # Remaining order retries (starts at 3)
    cook_retry_attempts: int       # Remaining cook retries (starts at 2)
    serve_retry_attempts: int      # Remaining serve retries (starts at 2)
    final_result: str              # "completed" or "not_completed"
```

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/shubh050307-hash/Restaurant-order-management-AI-agent.git
cd Restaurant-order-management-AI-agent

# Create virtual environment
uv venv

# Install dependencies
uv pip install langgraph langchain-groq langchain-core pydantic python-dotenv
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Run the interactive agent

```bash
.venv/bin/python main.py
```

**Example session:**

```
============================================================
  🍽️  Welcome to the AI Restaurant Order Agent!
  Menu:
    • Burger     — 10 available
    • Pizza      — 20 available
    • Momos      — 4 available
  Type 'exit' to quit.
============================================================

What would you like to order? I want 7 burgers

--- Order Transcript ---
👤 I want 7 burgers
🤖 Got it! You'd like to order 7 burger(s). Let me check availability...
🤖 Great news! burger is available. Proceeding to prepare your 7 burger(s).
🤖 Your 7 burger(s) have been cooked and are ready to serve!
🤖 🎉 Your order of 7 burger(s) has been served successfully! Enjoy your meal!

📋 Final Result: completed (status: complete)
```

### Run test scenarios

```bash
.venv/bin/python test_scenarios.py
```

This runs 3 automated test cases with mocked LLM, user input, and random outcomes:

| Test | Scenario | Expected |
|------|----------|----------|
| **TC1** | Unrelated question → partial order → reject → unavailable dishes → retry exhausted | ❌ not_completed |
| **TC2** | Available order → cook fail → cook ok → serve fail → cook ok → serve ok | ✅ completed |
| **TC3** | Partial → new order → cook fail → cook ok → serve fail → cook ok → serve fail → exhausted | ❌ not_completed |

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — State machine framework for the agent workflow
- **[Groq](https://groq.com/)** — LLM provider (model: `openai/gpt-oss-120b`)
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM integration layer
- **[Pydantic](https://docs.pydantic.dev/)** — Structured output parsing

## Project Structure

```
restproject/
├── main.py              # Agent code (nodes, routing, graph, CLI)
├── test_scenarios.py    # Automated test cases (mocked)
├── pyproject.toml       # Project metadata & dependencies
├── .env                 # API keys (not tracked in git)
├── .gitignore
├── .python-version
└── prompt.md            # Original requirements & test cases
```
