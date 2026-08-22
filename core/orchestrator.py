import re
import sys
import json
import operator
import logging
from typing import TypedDict, Annotated, List, Optional

from langgraph.graph import StateGraph, START, END

from tools.web_search import run_web_search
from tools.os_control import run_os_control
from sensory.vision import VisionPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Full conversation history. operator.add appends instead of overwriting.
    messages: Annotated[List[dict], operator.add]
    user_input: str
    llm_output: Optional[str]
    tool_result: Optional[str]
    screen_capture: Optional[str]
    memory_context: Optional[str]
    next_action: str            # "tool_call" | "respond" | "needs_approval"
    pending_approval: Optional[str]  # os_control command waiting for Y/N


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """
    Strip code fences LLMs love wrapping their output in.
    Handles ```json, ```python, and plain ``` blocks.
    Applied centrally in llm_node so every downstream function
    receives clean text - no duplication across tool files.
    """
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tiered approval helpers
# ---------------------------------------------------------------------------

# Read-only, non-destructive shell commands that are safe to auto-approve.
SAFE_COMMANDS = {
    "dir", "echo", "type", "ls", "pwd", "whoami",
    "hostname", "date", "time", "ver", "systeminfo",
    "ipconfig", "ping", "tracert", "netstat",
    "python --version", "pip list",
}


def _is_safe_command(command: str) -> bool:
    """Returns True if command starts with a known safe read-only prefix."""
    cmd_lower = command.strip().lower()
    return any(
        cmd_lower == safe or cmd_lower.startswith(safe + " ")
        for safe in SAFE_COMMANDS
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def vision_node(state: AgentState) -> dict:
    logger.info("[vision_node] capturing screen")
    try:
        pipeline = VisionPipeline()
        b64 = pipeline.get_vision_context()
        return {"screen_capture": b64}
    except Exception as e:
        logger.warning(f"[vision_node] screenshot failed: {e}")
        return {"screen_capture": None}


def memory_node(state: AgentState) -> dict:
    logger.info("[memory_node] retrieving context")
    # TODO: replace with real ChromaDB retrieval from memory/rag_memory.py
    return {"memory_context": None}


def llm_node(state: AgentState) -> dict:
    logger.info("[llm_node] querying LLM")
    # TODO: replace stub with real LLMEngine.generate() call.
    # Prompt assembly sketch:
    #   parts = [SYSTEM_PROMPT]
    #   if state["memory_context"]: parts.append(f"Memory:\n{state['memory_context']}")
    #   if state["tool_result"]: parts.append(f"Observation: {state['tool_result']}")
    #   for msg in state["messages"]: parts.append(f"{msg['role']}: {msg['content']}")
    #   parts.append(f"User: {state['user_input']}\nAssistant:")
    #   raw = engine.generate("\n\n".join(parts))

    # ---- Stub for smoke-testing ------------------------------------------
    raw = (
        '{"tool": "web_search", "input": "latest AI news"}'
        if state.get("tool_result") is None
        else f"Based on search results: {state['tool_result']}"
    )
    # ----------------------------------------------------------------------

    cleaned = strip_markdown(raw)
    is_tool_call = cleaned.startswith("{") and '"tool"' in cleaned

    return {
        "llm_output": cleaned,
        "next_action": "tool_call" if is_tool_call else "respond",
    }


def tool_node(state: AgentState) -> dict:
    raw = state["llm_output"]
    logger.info(f"[tool_node] dispatching: {raw[:80]}")

    # --- Parse JSON --------------------------------------------------------
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # LLM hallucinated non-JSON. Inject error as observation so the
        # model can self-correct on the next llm_node pass.
        logger.warning("[tool_node] JSON parse failed - asking LLM to retry")
        return {
            "tool_result": (
                'Error: your last response was not valid JSON. '
                'Output ONLY a raw JSON object with no backticks, '
                'e.g. {"tool": "web_search", "input": "your query"}'
            ),
            "next_action": "tool_call",
            "pending_approval": None,
        }

    tool_name = payload.get("tool", "").strip()
    tool_input = payload.get("input", "").strip()

    if tool_name not in ("web_search", "os_control"):
        return {
            "tool_result": f"Error: unknown tool '{tool_name}'. Available: web_search, os_control",
            "pending_approval": None,
        }

    if not tool_input:
        return {"tool_result": "Error: 'input' field is empty.", "pending_approval": None}

    # web_search: no approval ever needed
    if tool_name == "web_search":
        result = run_web_search(tool_input)
        logger.info(f"[tool_node] web_search result preview: {result[:100]}")
        return {"tool_result": result, "pending_approval": None}

    # os_control: tiered - safe commands auto-approve, risky ones halt for Y/N
    if tool_name == "os_control":
        if _is_safe_command(tool_input):
            logger.info(f"[tool_node] auto-approving safe command: {tool_input}")
            result = run_os_control(tool_input)
            logger.info(f"[tool_node] os_control result: {result[:100]}")
            return {"tool_result": result, "pending_approval": None}
        else:
            logger.info(f"[tool_node] queuing for human approval: {tool_input}")
            return {"pending_approval": tool_input, "next_action": "needs_approval"}


def approval_node(state: AgentState) -> dict:
    """
    Prints the pending os_control command and blocks on terminal input.
    Y  -> executes and injects result back into tool_result.
    N  -> injects a denial message so the LLM knows and can respond.
    Either way the graph loops back through llm_node to continue reasoning.
    """
    command = state["pending_approval"]

    print("\n" + "=" * 60)
    print("[JARVIS] Wants to run the following OS command:")
    print(f"  >>> {command}")
    print("=" * 60)

    while True:
        choice = input("Approve? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            logger.info("[approval_node] approved - executing")
            result = run_os_control(command)
            return {
                "tool_result": result,
                "pending_approval": None,
                "next_action": "tool_call",
            }
        elif choice in ("n", "no"):
            logger.info("[approval_node] denied by user")
            return {
                "tool_result": f"User denied execution of: {command}",
                "pending_approval": None,
                "next_action": "tool_call",
            }
        else:
            print("Please type Y or N.")


def respond_node(state: AgentState) -> dict:
    logger.info("[respond_node] finalising answer")
    new_message = {"role": "assistant", "content": state["llm_output"]}
    # Wipe scratchpad so stale data never leaks into the next turn
    return {
        "messages": [new_message],
        "screen_capture": None,
        "memory_context": None,
        "tool_result": None,
        "pending_approval": None,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    """Routes from llm_node: tool_call | needs_approval | respond."""
    return state["next_action"]


def approval_router(state: AgentState) -> str:
    """Routes from tool_node: straight to llm_node or to approval gate."""
    return state.get("next_action", "tool_call")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("vision_node", vision_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("approval_node", approval_node)
    graph.add_node("respond_node", respond_node)

    graph.add_edge(START, "vision_node")
    graph.add_edge("vision_node", "memory_node")
    graph.add_edge("memory_node", "llm_node")

    # llm_node: run tool | halt for approval | final answer
    graph.add_conditional_edges(
        "llm_node",
        router,
        {
            "tool_call": "tool_node",
            "needs_approval": "approval_node",
            "respond": "respond_node",
        },
    )

    # tool_node: safe/web commands go back to llm; risky ones go to approval
    graph.add_conditional_edges(
        "tool_node",
        approval_router,
        {
            "tool_call": "llm_node",
            "needs_approval": "approval_node",
        },
    )

    # After approval (Y or N), always loop back so LLM sees the outcome
    graph.add_edge("approval_node", "llm_node")
    graph.add_edge("respond_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_graph()

    # Default: web_search path - no approval needed, runs end-to-end silently.
    # Pass "os" as CLI arg to test the Y/N approval prompt:
    #   python -m core.orchestrator os
    test_os = len(sys.argv) > 1 and sys.argv[1] == "os"

    if test_os:
        print("=== Smoke test: os_control approval path ===")
        state = {
            "messages": [],
            "user_input": "List all files here",
            "llm_output": '{"tool": "os_control", "input": "Get-ChildItem"}',
            "tool_result": None,
            "screen_capture": None,
            "memory_context": None,
            "next_action": "tool_call",
            "pending_approval": None,
        }
    else:
        print("=== Smoke test: web_search path (no approval) ===")
        state = {
            "messages": [],
            "user_input": "What is the latest news in AI?",
            "llm_output": None,
            "tool_result": None,
            "screen_capture": None,
            "memory_context": None,
            "next_action": "",
            "pending_approval": None,
        }

    result = app.invoke(state)
    print("\n=== Final state ===")
    content = result["messages"][0]["content"][:200] if result["messages"] else "none"
    print("messages   :", content)
    print("next_action:", result["next_action"])
