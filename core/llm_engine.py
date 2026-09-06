import logging
from typing import List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt - kept minimal so the model's own training drives tool choice
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Augmented Jackdaw (AJ), a local AI assistant built by Parimarjan. You run fully on-device.

To use a tool, output ONLY a single JSON object - nothing else before or after it:
{"tool": "web_search", "input": "your search query"}
{"tool": "os_control", "input": "shell command or python script"}
{"tool": "save_memory", "input": "a fact worth remembering long-term"}

Otherwise respond in plain text.\
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Local inference wrapper for quantized GGUF models via llama-cpp-python.

    Init parameters:
      n_gpu_layers=-1  -> offload every layer to RTX 2050 VRAM
      n_ctx=8192       -> full context window (matches model training size)
      n_batch=512      -> tokens per batch
      f16_kv=True      -> float16 KV cache (~20% VRAM saving vs float32)
    """

    def __init__(self):
        self.model: Optional[Llama] = None

    def load_model(self) -> bool:
        if Llama is None:
            logger.error(
                "llama-cpp-python is not installed.\n"
                "pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
            return False

        if not config.MODEL_PATH.exists():
            logger.error(f"Model not found at: {config.MODEL_PATH}")
            return False

        # Check for trained LoRA adapter (GGUF format required by llama.cpp)
        adapter_dir = config.MODELS_DIR / "lora_adapter"
        lora_path = None

        if adapter_dir.exists():
            for candidate in ["adapter_model.gguf", "lora_adapter.gguf", "adapter.gguf"]:
                candidate_path = adapter_dir / candidate
                if candidate_path.exists():
                    lora_path = str(candidate_path)
                    break

            if not lora_path:
                peft_names = [f.name for f in adapter_dir.glob("adapter_model.*")]
                if peft_names:
                    logger.warning(
                        f"Found PEFT adapter in {adapter_dir} ({peft_names}), "
                        "but llama-cpp requires a .gguf adapter. Convert to GGUF to activate."
                    )

        logger.info(f"Loading model from {config.MODEL_PATH} ...")
        if lora_path:
            logger.info(f"Applying LoRA adapter from {lora_path} ...")

        try:
            self.model = Llama(
                model_path=str(config.MODEL_PATH),
                lora_path=lora_path,
                n_ctx=config.CONTEXT_WINDOW,
                n_gpu_layers=-1,
                n_batch=512,
                f16_kv=True,
                verbose=False,
            )
            if lora_path:
                logger.info(f"Model loaded into VRAM with LoRA adapter from {lora_path}.")
            else:
                logger.info("Model loaded into VRAM.")
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
        tool_call_count: int = 0,
        max_tool_calls: int = 4,
    ) -> str:
        """
        Assemble a Gemma instruction-tuned prompt.

        The final <start_of_turn>model tag is left UNCLOSED intentionally —
        that is the open slot the model writes into.
        """
        parts: List[str] = []

        # Replay prior conversation turns
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts.append(f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>")

        # Build current user turn with injected context
        user_content = SYSTEM_PROMPT

        if memory_context:
            user_content += f"\n\n[Memory]\n{memory_context}\n[/Memory]"

        if tool_result:
            remaining = max_tool_calls - tool_call_count
            attempts_str = f"{remaining} attempt{'s' if remaining != 1 else ''} remaining"
            user_content += (
                f"\n\n[Tool result — call {tool_call_count}/{max_tool_calls}, {attempts_str}]\n"
                f"{tool_result}\n[/Tool result]"
            )

        user_content += f"\n\n{user_input}"

        parts.append(f"<start_of_turn>user\n{user_content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")  # open slot — no closing tag

        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        temperature: float = 0.7,
    ) -> str:
        if self.model is None:
            logger.warning("Model not loaded. Call load_model() first.")
            return ""

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
