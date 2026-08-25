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
from core.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

# Single engine instance - loaded once at startup, reused across all invocations
_engine = LLMEngine()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]   # operator.add appends, not overwrites
    user_input: str
    llm_output: Optional[str]
    tool_result: Optional[str]
    screen_capture: Optional[str]
    memory_context: Optional[str]
    next_action: str                 # "tool_call" | "respond" | "needs_approval"
    pending_approval: Optional[str]  # os_control command waiting for Y/N


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """
    Strip code fences LLMs love wrapping their output in.
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

SAFE_COMMANDS = {
    "dir", "echo", "type", "ls", "pwd", "whoami",
    "hostname", "date", "time", "ver", "systeminfo",
    "ipconfig", "ping", "tracert", "netstat",
    "python --version", "pip list",
}


def _is_safe_command(command: str) -> bool:
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
    # TODO: wire in ChromaDB retrieval from memory/rag_memory.py
    return {"memory_context": None}


def llm_node(state: AgentState) -> dict:
    logger.info("[llm_node] querying LLM")

    # Lazy load - model is loaded on the first call, then reused
    if _engine.model is None:
        loaded = _engine.load_model()
        if not loaded:
            return {
                "llm_output": "Error: LLM model could not be loaded. Check model path and llama-cpp-python install.",
                "next_action": "respond",
            }

    prompt = _engine.build_prompt(
        user_input=state["user_input"],
        messages=state["messages"],
        memory_context=state.get("memory_context"),
        tool_result=state.get("tool_result"),
    )
    raw = _engine.generate(prompt)

    cleaned = strip_markdown(raw)
    is_tool_call = cleaned.startswith("{") and '"tool"' in cleaned

    return {
        "llm_output": cleaned,
        "next_action": "tool_call" if is_tool_call else "respond",
    }


def tool_node(state: AgentState) -> dict:
    raw = state["llm_output"]
    logger.info(f"[tool_node] dispatching: {raw[:80]}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Model hallucinated non-JSON - inject error as observation so it self-corrects
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

    # web_search - no approval ever needed
    if tool_name == "web_search":
        result = run_web_search(tool_input)
        logger.info(f"[tool_node] web_search result: {result[:100]}")
        return {"tool_result": result, "pending_approval": None}

    # os_control - tiered: safe commands auto-approve, risky ones halt for Y/N
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
    Y -> executes and injects stdout into tool_result.
    N -> injects a denial message so the LLM can respond accordingly.
    Either way the graph loops back through llm_node.
    """
    command = state["pending_approval"]
    print("\n" + "=" * 60)
    print("[AJ] Wants to run the following OS command:")
    print(f"  >>> {command}")
    print("=" * 60)

    while True:
        choice = input("Approve? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            logger.info("[approval_node] approved - executing")
            result = run_os_control(command)
            return {"tool_result": result, "pending_approval": None, "next_action": "tool_call"}
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
    """Routes from tool_node: to llm_node (most cases) or approval gate."""
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

    graph.add_conditional_edges(
        "llm_node",
        router,
        {"tool_call": "tool_node", "needs_approval": "approval_node", "respond": "respond_node"},
    )

    graph.add_conditional_edges(
        "tool_node",
        approval_router,
        {"tool_call": "llm_node", "needs_approval": "approval_node"},
    )

    graph.add_edge("approval_node", "llm_node")
    graph.add_edge("respond_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_interactive():
    """Simple CLI loop - type your prompt, AJ responds."""
    logging.basicConfig(level=logging.WARNING)  # suppress INFO noise in interactive mode
    app = build_graph()

    print("=== Augmented Jackdaw (AJ) - Local AI Agent ===")
    print("Type 'exit' to quit.\n")

    messages: List[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        state: AgentState = {
            "messages": messages,
            "user_input": user_input,
            "llm_output": None,
            "tool_result": None,
            "screen_capture": None,
            "memory_context": None,
            "next_action": "",
            "pending_approval": None,
        }

        result = app.invoke(state)

        # Update history with user turn + assistant reply
        messages.append({"role": "user", "content": user_input})
        if result["messages"]:
            reply = result["messages"][-1]["content"]
            messages.append({"role": "assistant", "content": reply})
            print(f"\nAJ: {reply}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Quick smoke test (no model needed - will fail gracefully)
        logging.basicConfig(level=logging.INFO)
        app = build_graph()
        result = app.invoke({
            "messages": [], "user_input": "Hello",
            "llm_output": None, "tool_result": None,
            "screen_capture": None, "memory_context": None,
            "next_action": "", "pending_approval": None,
        })
        print("Result:", result["messages"])
    else:
        run_interactive()
