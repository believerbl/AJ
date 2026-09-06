"""
Training Logger — "Correction & Capture" strategy.

Every interaction that flows through the LangGraph orchestrator gets logged
here in Gemma chat format (.jsonl). The key insight:

  We log the CORRECTED version of the output (what AJ *should* have said),
  not the raw hallucinated one. So if AJ outputs `_code {"tool": "web_search"}`
  the logger strips the prefix and writes clean {"tool": "web_search"} to disk.

After a few days of normal use, this file becomes the LoRA training dataset.
The idle_trainer.py will pick it up automatically when your laptop goes quiet.
"""

import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)

DATASET_PATH = config.DATA_DIR / "my_jarvis_data.jsonl"


def _gemma_format(system_prompt: str, user_input: str, aj_response: str) -> dict:
    """
    Formats an interaction in Gemma instruction-tuned chat format.
    This is the exact format the LoRA trainer expects.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{system_prompt}\n\n{user_input}"
            },
            {
                "role": "model",
                "content": aj_response
            }
        ]
    }


def log_interaction(
    user_input: str,
    aj_response: str,
    system_prompt: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[str] = None,
) -> None:
    """
    Log a single interaction to the training dataset.

    If the interaction involved a tool call, aj_response is the CORRECTED
    clean JSON (not the raw hallucinated output). This is how we bootstrap
    the LoRA dataset even when the base model outputs garbage prefixes.

    Args:
        user_input:    What the user said.
        aj_response:   AJ's clean final response (plain text or clean JSON).
        system_prompt: The system prompt used for this turn.
        tool_name:     If a tool was called, its name (for metadata only).
        tool_input:    If a tool was called, its input (for metadata only).
    """
    if not user_input.strip() or not aj_response.strip():
        return

    record = _gemma_format(system_prompt, user_input, aj_response)

    # Add metadata outside the training record (stripped before training)
    record["_meta"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool_name,
        "tool_input": tool_input,
        # Content hash to deduplicate identical interactions
        "id": hashlib.md5(
            f"{user_input}{aj_response}".encode("utf-8")
        ).hexdigest(),
    }

    try:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug(f"[training_logger] logged interaction (tool={tool_name})")
    except Exception as e:
        logger.error(f"[training_logger] failed to write: {e}")


def dataset_size() -> int:
    """Returns the number of logged interactions."""
    if not DATASET_PATH.exists():
        return 0
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
