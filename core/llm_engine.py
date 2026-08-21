import json
import logging
from typing import Dict, Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import config

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Local LLM inference wrapper using llama-cpp-python.
    Handles loading quantized GGUF models within VRAM constraints.
    """
    def __init__(self):
        self.model = None
        
    def load_model(self):
        if Llama is None:
            logger.error("llama-cpp-python is not installed. Please install it to use the local LLM.")
            return

        if not config.MODEL_PATH.exists():
            logger.warning(f"Model file not found at {config.MODEL_PATH}. Inference will be simulated or fail.")
            return

        logger.info(f"Loading model from {config.MODEL_PATH}")
        try:
            # Note: n_gpu_layers can be tuned based on VRAM limits.
            # -1 usually means offload all layers to GPU if possible.
            self.model = Llama(
                model_path=str(config.MODEL_PATH),
                n_ctx=config.CONTEXT_WINDOW,
                n_gpu_layers=-1, # Offload to GPU
                verbose=False
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def generate(self, prompt: str, max_tokens: int = 512, stop: Optional[list] = None) -> str:
        """
        Generate text response from the loaded model.
        """
        if self.model is None:
            logger.warning("Model is not loaded. Returning empty string.")
            return ""

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                stop=stop or ["Observation:", "\n\nUser:"],
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    def generate_json(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate text and parse it as JSON. Useful for tool calling.
        Expects the LLM to output valid JSON format.
        """
        # Append instruction for JSON format if not already in prompt
        json_prompt = prompt + "\n\nEnsure your response is valid JSON only."
        
        response_text = self.generate(json_prompt, max_tokens=max_tokens)
        
        try:
            # Basic cleanup in case the LLM wraps it in markdown code blocks
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM output: {response_text}\nError: {e}")
            return {"error": "Failed to parse JSON", "raw_output": response_text}
