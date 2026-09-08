import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MEMORY_DIR = BASE_DIR / "memory_db"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, MEMORY_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Hugging Face Cache Settings (store on project drive D: to prevent filling up C:)
HF_CACHE_DIR = MODELS_DIR / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# Hardware Limits & Configuration
VRAM_LIMIT_GB = 4.0
VRAM_BUFFER_GB = 0.5
USABLE_VRAM_GB = VRAM_LIMIT_GB - VRAM_BUFFER_GB

# LLM Engine Settings
MODEL_NAME = "gemma-2-2b-it-abliterated-Q4_K_M" # Placeholder for the local quantized model path or name
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.gguf" 
CONTEXT_WINDOW = 8192

# RAG Memory Settings
# Prefer self-contained local embedding weights (prevents offline HuggingFace lookup errors)
EMBEDDING_MODEL_PATH = MODELS_DIR / "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = str(EMBEDDING_MODEL_PATH) if EMBEDDING_MODEL_PATH.exists() else "all-MiniLM-L6-v2"
CHROMA_DB_DIR = str(MEMORY_DIR)

# Sensory (Vision) Settings
SCREENSHOT_RESIZE_FACTOR = 0.5 # Compress screenshots to save context
SCREENSHOT_MAX_SIZE = (1024, 768)

# Continuous Learning Settings
DATASET_PATH = DATA_DIR / "my_jarvis_data.jsonl"
IDLE_TIMEOUT_SECONDS = 600 # 10 minutes

# Training chunk settings
TRAINING_CHUNK_SIZE = 50        # minimum examples before a training run starts
TRAINING_CURSOR_FILE = DATA_DIR / "training_cursor.json"  # tracks consumed examples

# LoRA Training Settings
LORA_R = 128
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAINING_BATCH_SIZE = 1 # Keep low for VRAM limit
TRAINING_BASE_MODEL = os.getenv("TRAINING_BASE_MODEL", "failspy/gemma-2-2b-it-abliterated")