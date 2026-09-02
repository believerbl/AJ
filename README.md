# Augmented Jackdaw (AJ)

> A fully local, autonomous AI agent that runs entirely on your own hardware — no cloud, no API keys, no subscriptions.

AJ is a personal AI assistant built on a **LangGraph state machine**, powered by a quantized local LLM via `llama-cpp-python`. It can search the web, execute system commands, remember past conversations, and see your screen — all without a single byte leaving your machine.

---

## What it can do

| Capability | Module | Status |
|---|---|---|
| Answer questions (local LLM) | `core/llm_engine.py` | ✅ Ready (needs model file) |
| Real-time web search | `tools/web_search.py` | ✅ Live |
| Execute shell / Python scripts | `tools/os_control.py` | ✅ Live |
| Human-in-the-loop approval gate | `core/orchestrator.py` | ✅ Live |
| Long-term memory (RAG) | `memory/rag_memory.py` | 🔧 Wired, testing pending |
| Desktop vision | `sensory/vision.py` | 🔧 Wired, headless limitation |
| Continuous background learning | `learning/idle_trainer.py` | 🔧 Phase 6 |

---

## Architecture

AJ is built as a **LangGraph ReAct loop** — a cyclical state machine where the LLM reasons, calls tools, observes results, and reasons again until it has a final answer.

```
START
  │
  ▼
vision_node        ← captures desktop screenshot
  │
  ▼
memory_node        ← retrieves relevant past context from ChromaDB
  │
  ▼
llm_node           ← sends assembled Gemma prompt to local model
  │
  ├──"tool_call"──► tool_node ──────────────────────► llm_node (loop)
  │                    │
  │              "needs_approval"
  │                    │
  │               approval_node  ← Y/N terminal prompt for risky commands
  │                    │
  │                    └──────────────────────────────► llm_node (loop)
  │
  └──"respond"──► respond_node ──► END
```

### Key design decisions

1. **Single-file constraint until 400 lines** — modules stay in one file until they grow past that, keeping the codebase readable without over-engineering.
2. **Flat JSON tool schema** — `{"tool": "web_search", "input": "..."}` — intentionally minimal so a 2B model can reliably produce it without hallucinating nested structures.
3. **Central `strip_markdown()`** in `llm_node` — strips code fences from the LLM output once, before anything else sees the text, instead of duplicating the logic in every tool.
4. **Tiered approval for `os_control`** — read-only commands (`dir`, `echo`, `ls`, etc.) auto-approve; anything else halts and asks for `Y/N` in the terminal.
5. **Self-healing JSON errors** — if the LLM outputs invalid JSON, the error is injected back as a tool observation so the model corrects itself on the next pass, no crash.

---

## Project structure

```
AJ/
├── core/
│   ├── llm_engine.py       # Llama.cpp wrapper, Gemma prompt builder, GPU config
│   └── orchestrator.py     # LangGraph graph: nodes, edges, routers, CLI loop
├── tools/
│   ├── web_search.py       # DuckDuckGo search via ddgs
│   └── os_control.py       # subprocess shell/Python execution with safety gate
├── memory/
│   └── rag_memory.py       # ChromaDB vector store for long-term memory
├── sensory/
│   └── vision.py           # Desktop screenshot capture and compression
├── learning/
│   ├── data_collector.py   # Logs interactions to JSONL for training
│   └── idle_trainer.py     # LoRA fine-tuning during idle periods
├── config.py               # All paths, VRAM limits, and hyperparameters
├── main.py                 # Alternative entry point
└── requirements.txt
```

---

## Hardware requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA GPU with CUDA support (tested on RTX 2050, 4 GB VRAM) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~3 GB for model + dependencies |
| OS | Windows 11 (WSL2 supported) |

---

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/believerbl/AJ.git
cd AJ
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** `llama-cpp-python` requires a C++ compiler to build from source on Windows.
> Skip the compile entirely by installing the pre-built CUDA wheel instead:
> ```powershell
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
> ```

### 3. Download the model

Download a 4-bit quantized GGUF of **Gemma 2B-IT** (the `Q4_K_M` variant is recommended):

- 👉 [bartowski/gemma-2-2b-it-GGUF on HuggingFace](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF)

Place the downloaded file here:
```
AJ/models/gemma-2b-it-GGUF.gguf
```

The model path is configured in `config.py` — edit `MODEL_NAME` if you use a different filename.

---

## Running AJ

```powershell
.\venv\Scripts\activate
python -m core.orchestrator
```

You will see an interactive prompt:
```
=== Augmented Jackdaw (AJ) - Local AI Agent ===
Type 'exit' to quit.

You: What is the weather in Delhi today?
```

AJ will reason, call web search, and reply — all locally.

### Test the OS approval gate

```powershell
python -m core.orchestrator os
```

This triggers a fake `Get-ChildItem` command so you can see the `[Y/n]` approval prompt before any real system access happens.

---

## Safety

AJ has a **two-tier safety model** for OS commands:

- **Auto-approved** (read-only, no side effects): `dir`, `echo`, `ls`, `ping`, `whoami`, `ipconfig`, `pip list`, etc.
- **Human approval required** (everything else): AJ prints the exact command it wants to run and waits for your explicit `Y` or `N` before touching anything.

Web searches always run without approval since they have no write access to your system.

---

## Configuration

All tunable parameters live in [`config.py`](config.py):

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gemma-2b-it-GGUF` | GGUF filename (without `.gguf`) |
| `CONTEXT_WINDOW` | `4096` | LLM context size in tokens |
| `IDLE_TIMEOUT_SECONDS` | `600` | Idle time before background training kicks in |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | CPU embedding model for RAG memory |

---

## Roadmap

- [x] Phase 1 — Project scaffold, LLM engine, vision pipeline
- [x] Phase 2 — LangGraph orchestrator skeleton with AgentState
- [x] Phase 3 — Web search + OS control tools with safety gate
- [x] Phase 4 — Human-in-the-Loop approval node
- [x] Phase 5 — Real LLM integration (Gemma prompt template + GPU offloading)
- [ ] Phase 6 — ChromaDB long-term memory (RAG)
- [ ] Phase 7 — Idle-time LoRA fine-tuning loop

---

*Built as a personal local AI engineering project. No cloud. No tracking. No cost.*
