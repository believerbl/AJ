"""
AJ LoRA Trainer — Phase 7 Engine

QLoRA fine-tuning on the accumulated interaction data in data/my_jarvis_data.jsonl.
Designed for a 4 GB VRAM budget (RTX 2050) using 4-bit quantization + PEFT LoRA.

Chunk system:
  - Reads exactly TRAINING_CHUNK_SIZE new examples (past the stored cursor).
  - Refuses to start if fewer than one full chunk is available.
  - Advances the cursor file after a successful run so examples are never
    trained on twice.

The base model is loaded from HuggingFace (not the GGUF file — GGUF cannot be
fine-tuned directly). The LoRA adapter is saved to models/lora_adapter/.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config

# Training log — idle_monitor pipes stdout here, so use print() for progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoRATrainer")


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _read_cursor() -> int:
    path = config.TRAINING_CURSOR_FILE
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("cursor", 0)
    except Exception:
        return 0


def _write_cursor(new_cursor: int):
    config.TRAINING_CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRAINING_CURSOR_FILE, "w", encoding="utf-8") as f:
        json.dump({"cursor": new_cursor, "last_trained": datetime.utcnow().isoformat()}, f)


def load_chunk() -> list[dict]:
    """
    Read exactly TRAINING_CHUNK_SIZE examples starting from the current cursor.
    Returns an empty list if fewer than one full chunk is available.
    """
    path = config.DATASET_PATH
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        return []

    cursor = _read_cursor()
    chunk_size = config.TRAINING_CHUNK_SIZE
    chunk = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < cursor:
                continue                  # skip already-consumed examples
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Strip _meta before handing to trainer
                record.pop("_meta", None)
                chunk.append(record)
            except json.JSONDecodeError:
                continue
            if len(chunk) >= chunk_size:
                break

    if len(chunk) < chunk_size:
        logger.info(
            f"Only {len(chunk)} new examples available "
            f"(need {chunk_size}). Aborting — will retry next idle window."
        )
        return []

    logger.info(f"Loaded chunk: examples {cursor} → {cursor + len(chunk)}")
    return chunk


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training():
    logger.info("=" * 55)
    logger.info("AJ LoRA Training Run")
    logger.info(f"Chunk size : {config.TRAINING_CHUNK_SIZE}")
    logger.info(f"LoRA r     : {config.LORA_R}")
    logger.info("=" * 55)

    # ── 1. Load chunk ────────────────────────────────────────────────────────
    chunk = load_chunk()
    if not chunk:
        logger.info("No full chunk available. Exiting.")
        sys.exit(0)

    cursor_before = _read_cursor()

    # ── 2. Lazy-import heavy deps (only loaded when training actually starts) ─
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error(
            f"Missing training dependency: {e}\n"
            "Install with: pip install peft transformers datasets bitsandbytes accelerate"
        )
        sys.exit(1)

    # Pre-flight check: CUDA required for 4-bit quantization on RTX 2050
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available in PyTorch. 4-bit QLoRA training requires an NVIDIA GPU with CUDA.\n"
            "To install CUDA-enabled PyTorch in your virtual environment, run:\n"
            "  uv pip install --python .\\venv\\Scripts\\python.exe --reinstall torch --index https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)

    # ── 3. Model & tokenizer ─────────────────────────────────────────────────
    # Configurable base model (defaults to unsloth/gemma-2-2b-it for unauthenticated access)
    BASE_MODEL = getattr(config, "TRAINING_BASE_MODEL", "unsloth/gemma-2-2b-it")
    ADAPTER_DIR = config.MODELS_DIR / "lora_adapter"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    logger.info(f"Loading base model: {BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",          # auto-places on RTX 2050
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    # ── 4. LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config.LORA_R,            # rank (128 from config)
        lora_alpha=config.LORA_R * 2,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Tokenise chunk ────────────────────────────────────────────────────
    def format_example(example: dict) -> str:
        """Convert a messages record back into a Gemma prompt string."""
        parts = []
        for msg in example.get("messages", []):
            role = "user" if msg["role"] == "user" else "model"
            parts.append(f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>")
        return "\n".join(parts)

    texts = [format_example(ex) for ex in chunk]
    tokenised = tokenizer(
        texts,
        truncation=True,
        max_length=config.CONTEXT_WINDOW,
        padding="max_length",
        return_tensors="pt",
    )
    tokenised["labels"] = tokenised["input_ids"].clone()

    dataset = Dataset.from_dict({k: v.tolist() for k, v in tokenised.items()})

    # ── 6. TrainingArguments — tuned for 4 GB VRAM ──────────────────────────
    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        per_device_train_batch_size=1,       # VRAM budget: 1 sample at a time
        gradient_accumulation_steps=8,       # effective batch = 8
        num_train_epochs=1,                  # one pass per chunk
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="no",                  # save manually after run
        optim="paged_adamw_8bit",            # 8-bit optimizer saves ~1 GB VRAM
        report_to="none",
    )

    # ── 7. Train ─────────────────────────────────────────────────────────────
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    logger.info("Training started...")
    trainer.train()
    logger.info("Training complete.")

    # ── 8. Save adapter + advance cursor ─────────────────────────────────────
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    logger.info(f"LoRA adapter saved to {ADAPTER_DIR}")

    new_cursor = cursor_before + len(chunk)
    _write_cursor(new_cursor)
    logger.info(f"Cursor advanced: {cursor_before} → {new_cursor}")
    logger.info("Done. VRAM will be released when this process exits.")


if __name__ == "__main__":
    run_training()
