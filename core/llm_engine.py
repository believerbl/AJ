import logging
from typing import List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Augmented Jackdaw (AJ), a local autonomous AI assistant running
entirely on-device with no cloud dependency.

When you need real-time information or to execute a system command,
respond with ONLY a raw JSON object on a single line - no backticks,
no explanation, nothing else:
  {"tool": "web_search", "input": "your search query"}
  {"tool": "os_control", "input": "your shell command or python script"}

When you have enough information to answer the user directly,
respond in plain text with no JSON.

Available tools: web_search, os_control\
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Local inference wrapper for quantized GGUF models via llama-cpp-python.

    Key init parameters explained:
      n_gpu_layers=-1  -> offload every transformer layer to the RTX 2050 VRAM.
                          For a 4-bit quantized 2B model (~1.5 GB) this fits
                          entirely in the 4 GB VRAM budget.
      n_ctx=4096       -> context window sized to comfortably hold system
                          prompt + tool results + conversation history
                          without hitting OOM.
      n_batch=512      -> tokens per batch - sweet spot for throughput on
                          a consumer GPU without stalling.
      f16_kv=True      -> float16 KV cache saves ~20% VRAM vs float32.
    """

    def __init__(self):
        self.model: Optional[Llama] = None

    def load_model(self) -> bool:
        """
        Load the GGUF model from disk into VRAM.
        Returns True on success, False on failure (so the caller can decide
        whether to abort or fall back gracefully).
        """
        if Llama is None:
            logger.error(
                "llama-cpp-python is not installed.\n"
                "Install the pre-built CUDA wheel with:\n"
                "  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
            return False

        if not config.MODEL_PATH.exists():
            logger.error(
                f"Model file not found at: {config.MODEL_PATH}\n"
                "Download a 4-bit quantized GGUF of Gemma 2B-IT and place it there."
            )
            return False

        logger.info(f"Loading model from {config.MODEL_PATH} ...")
        try:
            self.model = Llama(
                model_path=str(config.MODEL_PATH),
                n_ctx=config.CONTEXT_WINDOW,   # 4096 tokens
                n_gpu_layers=-1,                # all layers to RTX 2050
                n_batch=512,                    # batch size for throughput
                f16_kv=True,                    # float16 KV cache
                verbose=False,
            )
            logger.info("Model loaded successfully into VRAM.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def build_prompt(
        self,
        user_input: str,
        messages: List[dict],
        memory_context: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> str:
        """
        Assemble a Gemma-format prompt from the conversation state.

        Gemma instruction-tuned models expect strictly alternating turns:
            <start_of_turn>user
            [content]<end_of_turn>
            <start_of_turn>model
            [content]<end_of_turn>
            ...

        The final <start_of_turn>model is left UNCLOSED - that is the open
        completion slot the model writes into. Adding <end_of_turn> there
        would cause the model to generate nothing.
        """
        parts: List[str] = []

        # Replay prior turns from conversation history
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts.append(
                f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>"
            )

        # Build the current user turn, injecting all available context
        user_content = SYSTEM_PROMPT

        if memory_context:
            user_content += f"\n\nRelevant memory from past sessions:\n{memory_context}"

        if tool_result:
            user_content += f"\n\nObservation from last tool call:\n{tool_result}"

        user_content += f"\n\n{user_input}"

        parts.append(f"<start_of_turn>user\n{user_content}<end_of_turn>")

        # Open model completion slot - no closing tag intentionally
        parts.append("<start_of_turn>model\n")

        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Run inference and return the raw generated text.
        Stop sequences prevent the model from hallucinating extra turns.
        """
        if self.model is None:
            logger.warning("Model is not loaded. Call load_model() first.")
            return ""

        # These stop sequences prevent the model from generating fake
        # "User:" or "Observation:" turns after it finishes responding.
        default_stop = ["<end_of_turn>", "\n\nUser:", "\n\nObservation:"]

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                stop=stop or default_stop,
                temperature=temperature,
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
