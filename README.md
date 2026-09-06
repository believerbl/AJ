# Augmented Jackdaw (AJ)

> A fully local, autonomous, self-improving AI agent engineered for consumer hardware. Zero cloud, zero API keys, zero subscriptions — 100% private.

Augmented Jackdaw is a personal AI assistant built on a **LangGraph state machine** and powered by local quantized LLMs via `llama-cpp-python`. It searches the web, executes system commands, recalls long-term memories via local vector RAG, observes your screen, and **autonomously fine-tunes itself on its own interactions while your computer is idle** — all engineered to run within a strict **4 GB VRAM budget** (NVIDIA RTX 2050).

---

## Architecture Overview

AJ operates on a dual-loop cognitive architecture: an **Interactive LangGraph ReAct Loop** for real-time task execution, and a background **Autonomous Delta-Chunk QLoRA Loop** for continuous self-improvement.

```
                           +----------------------------------------+
                           |          User Speaks to AJ             |
                           +-------------------+--------------------+
                                               |
                                               v
+========================================================================================+
|                        INTERACTIVE REACT LOOP (LangGraph Engine)                       |
|                                                                                        |
|    +---------------+        +---------------+        +----------------------------+    |
|    |  vision_node  | -----> |  memory_node  | -----> |          llm_node          |    |
|    | (Screenshots) |        | (ChromaDB RAG)|        | (Gemma-2B + LoRA in VRAM)  |    |
|    +---------------+        +---------------+        +--------------+-------------+    |
|                                                                     |                  |
|                                               +---------------------+----------------+ |
|                                               |                                      | |
|                                       "tool_call"                                "respond"
|                                               |                                      | |
|                                               v                                      v |
|                              +---------------------------------+             +-------+-+
|                              |   Regex JSON Fence Sanitizer    |             | respond |
|                              +----------------+----------------+             |  _node  |
|                                               |                              +----+----+
|                         +---------------------+---------------------+             |    |
|                         |                                           |             |    |
|               Safe / Read-Only Tool                           State-Changing Tool |    |
|                         |                                           |             |    |
|                         v                                           v             |    |
|                  +--------------+                         +-------------------+   |    |
|                  |  tool_node   |                         |   approval_node   |   |    |
|                  | (Web/OS/RAG) |                         | (Interactive Y/N) |   |    |
|                  +-------+------+                         +---------+---------+   |    |
|                          |                                          |             |    |
|                          +--------------------+---------------------+             |    |
|                                               |                                   |    |
|                                               v                                   |    |
|                                   [Observations & Attempts]                      |    |
|                                               |                                   |    |
|                                               +-----------------------------------+    |
+===============================================|========================================+
                                                |
                                                | Logs sanitized, perfected turns
                                                v
+========================================================================================+
|                    AUTONOMOUS CONTINUOUS LEARNING LOOP (Phase 7)                       |
|                                                                                        |
|         +--------------------------------------------------------------------+         |
|         |                   data/my_jarvis_data.jsonl                        |         |
|         |          (Self-healing data pipeline of gold-standard turns)       |         |
|         +---------------------------------+----------------------------------+         |
|                                           |                                            |
|                                           v                                            |
|         +--------------------------------------------------------------------+         |
|         |                Chunk Cursor Tracking (50 items/chunk)              |         |
|         |     Ensures training only runs on new deltas -- never grows stale  |         |
|         +---------------------------------+----------------------------------+         |
|                                           |                                            |
|                  +------------------------+------------------------+                   |
|                  |                                                 |                   |
|       Fewer than 50 new turns                              50+ turns & idle 10m        |
|                  |                                                 |                   |
|                  v                                                 v                   |
|         [Idle Watchdog Sleeps]                       +---------------------------+     |
|                                                      |  core/idle_monitor.py     |     |
|                                                      | (Windows GetLastInputInfo)|     |
|                                                      +-------------+-------------+     |
|                                                                    |                   |
|                                                                    v                   |
|                                                      +---------------------------+     |
|                                                      |    core/train_lora.py     |     |
|                                                      | (4-bit NF4 + 8-bit AdamW) |     |
|                                                      +-------------+-------------+     |
|                                                                    |                   |
|                      User moves mouse / presses key                | Training complete |
|                      ----------------------------->                v                   |
|                      Instant SIGTERM: VRAM freed             models/lora_adapter/      |
|                      Cursor intact for next idle                   |                   |
|                                                                    |                   |
|                                              Snaps weights on boot v                   |
|                                              +-----------------------------------+     |
|                                              |     core/llm_engine.py (Boot)     |     |
|                                              |  `lora_path` auto-applied to VRAM |     |
|                                              +-----------------------------------+     |
+========================================================================================+
```

---

## Capabilities & Feature Matrix

| Capability | Implementation | Hardware Strategy | Status |
|---|---|---|---|
| **Local LLM Engine** | `core/llm_engine.py` | Full GPU offload via `llama-cpp-python`, f16 KV cache, 8192 context window | Live |
| **Real-Time Web Search** | `tools/web_search.py` | DuckDuckGo search (`ddgs`) with smart retry & exception recovery | Live |
| **OS Control & Scripting** | `tools/os_control.py` | Safe subprocess execution in PowerShell/CMD/Python | Live |
| **Human-in-the-Loop Safety**| `core/orchestrator.py` | Two-tier gate: read-only auto-executes; destructive commands require `[Y/n]` | Live |
| **Long-Term Memory (RAG)** | `memory/rag_memory.py` | ChromaDB vector store running CPU-bound `all-MiniLM-L6-v2` embeddings | Live |
| **Desktop Vision** | `sensory/vision.py` | Downsampled screenshot capture with PIL compression | Live |
| **Self-Healing Output** | `core/orchestrator.py` | Regex fence stripping + brace-balanced JSON extractor + remaining attempt budget | Live |
| **Autonomous Watchdog** | `core/idle_monitor.py` | Native Windows API inactivity tracking; zero CPU/GPU overhead while active | Live |
| **Delta-Chunk QLoRA** | `core/train_lora.py` | 4-bit BitsAndBytes NF4 quantization + 8-bit Paged AdamW fitted to 4GB VRAM | Live |
| **Dynamic Adapter Snap** | `core/llm_engine.py` | Auto-detects and mounts trained GGUF LoRA adapter on startup via `lora_path` | Live |

---

## Key Engineering Decisions

1. **Delta-Chunk Training Cursor**: Rather than retraining on an ever-growing dataset (which causes exponential slowdowns), AJ uses a persistent cursor (`data/training_cursor.json`). It trains strictly on the newest delta chunk (50 interactions), advances the cursor, and skips already learned interactions.
2. **Zero-Latency Resource Preemption**: The background idle watchdog monitors native Windows user input (`GetLastInputInfo`). The exact millisecond you touch your mouse or keyboard, any active training process is sent `SIGTERM`, instantly freeing the GPU before you notice any frame drops.
3. **Regex Gatekeeper & Budget Pattern**: Smaller 2B models can struggle with strict JSON schemas. AJ strips outer markdown code fences (`^```[a-zA-Z]*` and `\n?```$`), extracts balanced JSON objects, and injects remaining-attempt budgets (`call 1/4, 3 attempts remaining`) instead of negative imperatives.
4. **Isolated Adapter Architecture**: Inference runs via fast GGUF C++ binaries, while fine-tuning runs in Python via HuggingFace PEFT/BitsAndBytes. The runtime cleanly checks for GGUF adapters and attaches them without modifying the base model weights on disk.

---

## Project Structure

```
AJ/
├── config.py                 # Central configurations (paths, VRAM limits, hyperparameters)
├── core/
│   ├── llm_engine.py         # Llama-cpp engine, prompt builder, LoRA auto-loader
│   ├── orchestrator.py       # LangGraph state machine, tool routing, safety gate
│   ├── training_logger.py    # Curates and appends perfected turns to JSONL
│   ├── idle_monitor.py       # Native Windows API idle watchdog & process supervisor
│   └── train_lora.py         # 4-bit QLoRA trainer with cursor tracking (VRAM-safe)
├── tools/
│   ├── web_search.py         # DuckDuckGo search integration
│   └── os_control.py         # Shell command execution & validation
├── memory/
│   └── rag_memory.py         # ChromaDB long-term memory store
├── sensory/
│   └── vision.py             # Desktop capture & image compression
├── data/
│   ├── my_jarvis_data.jsonl  # Accumulated training dataset (created at runtime)
│   └── training_cursor.json  # Delta chunk pointer (created at runtime)
├── models/                   # GGUF models and trained LoRA adapters
├── memory_db/                # ChromaDB persistent vector database
└── requirements.txt
```

---

## Hardware Requirements

| Component | Minimum Specification | Tested Specification |
|---|---|---|
| **GPU** | NVIDIA GPU with 4 GB VRAM (CUDA support) | NVIDIA GeForce RTX 2050 (4 GB) |
| **RAM** | 8 GB System Memory | 16 GB DDR4 |
| **Storage** | ~4 GB for model weights & dependencies | SSD |
| **OS** | Windows 10/11 64-bit | Windows 11 Home |

---

## Installation & Setup

### 1. Clone & Set Up Virtual Environment

```powershell
git clone https://github.com/believerbl/AJ.git
cd AJ
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Install the pre-compiled CUDA wheel for `llama-cpp-python` (matches your CUDA version):
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 3. Model Setup

Place your quantized `.gguf` file inside the `models/` directory:
```
models/gemma-2-2b-it-abliterated-Q4_K_M.gguf
```
*(Filename can be adjusted via `MODEL_NAME` in `config.py`)*

---

## Running Augmented Jackdaw

For the full autonomous experience, run the interactive agent and the idle watchdog in two terminal windows:

### Terminal 1 — The Interactive Agent
```powershell
cd D:\believer\codes\projects\AJ
.\venv\Scripts\activate
python -m core.orchestrator
```

### Terminal 2 — The Autonomous Idle Watchdog
```powershell
cd D:\believer\codes\projects\AJ
.\venv\Scripts\activate
python -m core.idle_monitor
```

---

## Configuration Reference

All hyperparameters are centralized in [`config.py`](config.py):

| Parameter | Default | Purpose |
|---|---|---|
| `VRAM_LIMIT_GB` | `4.0` | Total hardware VRAM ceiling |
| `CONTEXT_WINDOW` | `8192` | Model context window in tokens |
| `TRAINING_CHUNK_SIZE`| `50` | Number of interactions required before fine-tuning |
| `IDLE_TIMEOUT_SECONDS`| `600` | Inactivity delay (10 min) before triggering training |
| `LORA_R` | `128` | LoRA rank for adapter capacity |
| `TRAINING_BATCH_SIZE`| `1` | Batch size optimized for 4GB VRAM safety |

---

## Roadmap & Milestones

- [x] **Phase 1: Foundations** - Project scaffold, GGUF inference engine, vision capture.
- [x] **Phase 2: Graph Architecture** - LangGraph state machine with cyclic ReAct pattern.
- [x] **Phase 3: Core Tooling** - DuckDuckGo web search & sandboxed OS command executor.
- [x] **Phase 4: Safety Architecture** - Human-in-the-loop approval gate for non-read-only commands.
- [x] **Phase 5: Native Reasoning** - Gemma prompt assembly, GPU offload, balanced JSON extraction.
- [x] **Phase 6: Long-Term Memory** - ChromaDB vector store with contextual RAG recall.
- [x] **Phase 7: Autonomous Self-Training** - Background watchdog, delta chunk cursor, 4-bit QLoRA trainer, and dynamic LoRA adapter loading.

---

*Augmented Jackdaw (AJ) is an independent local AI engineering project built by Parimarjan. Completely private, fully local, and self-improving.*
