import json
import logging
import threading

import config

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Logs conversational pairs into a JSONL file for continuous LoRA training.
    """
    def __init__(self):
        self.dataset_path = config.DATASET_PATH
        self.lock = threading.Lock()

    def log_interaction(self, system_prompt: str, user_prompt: str, ai_response: str):
        """
        Appends an interaction to the JSONL dataset.
        Formats it in a standard instruction-tuning format (e.g., Alpaca/ChatML).
        """
        # Create a single dictionary representing the chat conversation
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": ai_response}
            ]
        }
        
        try:
            # Use lock to prevent concurrent write issues if async tasks overlap
            with self.lock:
                with open(self.dataset_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
            logger.debug("Logged interaction for future training.")
        except Exception as e:
            logger.error(f"Failed to log interaction to {self.dataset_path}: {e}")
