import re
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
    next_action: str   # "tool_call" | "respond"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """
    Strip code fences that LLMs love wrapping their output in.
    Handles ```json, ```python, and plain ``` blocks.
    Applied centrally in llm_node so every downstream function
    receives clean text � no duplication across tool files.
    """
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def vision_node(state: AgentState) -> dict:
    logger.info("[vision_node] capturing screen")
    try:
        pipeline = VisionPipeline()
        b64 = pipeline.get_screen_base64()
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
    # The prompt assembly will look roughly like:
    #
    #   prompt_parts = [SYSTEM_PROMPT]
    #   if state["memory_context"]:
    #       prompt_parts.append(f"Relevant memory:\n{state['memory_context']}")
    #   if state["tool_result"]:
    #       prompt_parts.append(f"Observation: {state['tool_result']}")
    #   for msg in state["messages"]:
    #       prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    #   prompt_parts.append(f"User: {state['user_input']}\nAssistant:")
    #   raw = engine.generate("\n\n".join(prompt_parts))

    # ---- Stub for testing ------------------------------------------------
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

    # --- Parse JSON ---------------------------------------------------------
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # LLM hallucinated non-JSON. Inject the error as an observation so
        # the model can self-correct on the next llm_node pass.
        logger.warning("[tool_node] JSON parse failed � asking LLM to retry")
        return {
            "tool_result": (
                'Error: your last response was not valid JSON. '
                'Output ONLY a raw JSON object with no backticks, '
                'e.g. {"tool": "web_search", "input": "your query"}'
            ),
            "next_action": "tool_call",
        }

    tool_name = payload.get("tool", "").strip()
    tool_input = payload.get("input", "").strip()

    # --- Dispatch -----------------------------------------------------------
    TOOLS = {
        "web_search": run_web_search,
        "os_control": run_os_control,
    }

    if tool_name not in TOOLS:
        result = f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
    elif not tool_input:
        result = "Error: 'input' field is empty."
    else:
        result = TOOLS[tool_name](tool_input)

    logger.info(f"[tool_node] result preview: {result[:100]}")
    return {"tool_result": result}


def respond_node(state: AgentState) -> dict:
    logger.info("[respond_node] finalising answer")
    new_message = {"role": "assistant", "content": state["llm_output"]}
    # Wipe scratchpad so stale data never leaks into the next turn
    return {
        "messages": [new_message],
        "screen_capture": None,
        "memory_context": None,
        "tool_result": None,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    return state["next_action"]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("vision_node", vision_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("respond_node", respond_node)

    graph.add_edge(START, "vision_node")
    graph.add_edge("vision_node", "memory_node")
    graph.add_edge("memory_node", "llm_node")

    graph.add_conditional_edges(
        "llm_node",
        router,
        {"tool_call": "tool_node", "respond": "respond_node"},
    )

    # After a tool runs, loop back for the LLM to reason over the result
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("respond_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_graph()

    initial_state: AgentState = {
        "messages": [],
        "user_input": "What is the latest news in AI?",
        "llm_output": None,
        "tool_result": None,
        "screen_capture": None,
        "memory_context": None,
        "next_action": "",
    }

    print("=== Running Jarvis orchestrator ===")
    result = app.invoke(initial_state)
    print("\n=== Final state ===")
    print("messages   :", result["messages"])
    print("next_action:", result["next_action"])
