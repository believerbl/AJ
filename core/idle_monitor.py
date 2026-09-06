"""
AJ Idle Monitor — Phase 7 Watchdog

Monitors keyboard/mouse inactivity using native Windows APIs (zero extra deps).
When the system has been idle for `idle_threshold` seconds AND a full chunk of
training data is available, it launches core/train_lora.py in a subprocess.
The instant the user touches the mouse or keyboard, the training process is
terminated so the RTX 2050's VRAM is freed immediately.

Run in a separate terminal:
    python -m core.idle_monitor
"""

import json
import time
import ctypes
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import config

# ---------------------------------------------------------------------------
# Logging — all output goes to logs/training_process.log
# ---------------------------------------------------------------------------

LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_LOG = LOG_DIR / "training_process.log"

logging.basicConfig(
    filename=str(TRAINING_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("IdleMonitor")

# Also print key events to terminal so you can see it is alive
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
logger.addHandler(_console)


# ---------------------------------------------------------------------------
# Windows idle time via GetLastInputInfo
# ---------------------------------------------------------------------------

class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_idle_seconds() -> float:
    """Returns how long the system has been idle (no keyboard/mouse) in seconds."""
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        elapsed_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
        return elapsed_ms / 1000.0
    return 0.0


# ---------------------------------------------------------------------------
# Chunk readiness check
# ---------------------------------------------------------------------------

def _count_total_examples() -> int:
    """Count lines in the JSONL training file."""
    path = config.DATASET_PATH
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _get_cursor() -> int:
    """Read the training cursor — how many examples have already been consumed."""
    path = config.TRAINING_CURSOR_FILE
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("cursor", 0)
    except Exception:
        return 0


def new_examples_available() -> tuple[bool, int]:
    """
    Returns (ready, count) where ready is True when at least one full chunk
    of TRAINING_CHUNK_SIZE new (unconsumed) examples exists.
    """
    total = _count_total_examples()
    cursor = _get_cursor()
    new = total - cursor
    ready = new >= config.TRAINING_CHUNK_SIZE
    return ready, new


# ---------------------------------------------------------------------------
# Main watchdog loop
# ---------------------------------------------------------------------------

def start_monitor(idle_threshold: int = 300):
    """
    Poll every 2 seconds. Start training when idle + chunk ready.
    Kill training the instant the user returns.
    """
    logger.info(f"AJ Idle Monitor started | threshold={idle_threshold}s | chunk={config.TRAINING_CHUNK_SIZE}")
    print(f"\n[*] Training log → {TRAINING_LOG}")

    training_proc: subprocess.Popen | None = None

    try:
        while True:
            idle_sec = get_idle_seconds()
            ready, new_count = new_examples_available()

            # ── START training ───────────────────────────────────────────────
            if idle_sec >= idle_threshold and training_proc is None:
                if ready:
                    logger.info(
                        f"Idle {idle_sec:.0f}s | {new_count} new examples "
                        f"(≥ chunk size {config.TRAINING_CHUNK_SIZE}) → launching trainer"
                    )
                    with open(TRAINING_LOG, "a") as lf:
                        lf.write(f"\n{'='*55}\n")
                        lf.write(f"TRAINING RUN — {datetime.now().isoformat()}\n")
                        lf.write(f"New examples this chunk: {new_count}\n")
                        lf.write(f"{'='*55}\n")
                    training_proc = subprocess.Popen(
                        ["python", "-m", "core.train_lora"],
                        stdout=open(TRAINING_LOG, "a"),
                        stderr=subprocess.STDOUT,
                    )
                else:
                    logger.info(
                        f"Idle {idle_sec:.0f}s but only {new_count}/{config.TRAINING_CHUNK_SIZE} "
                        "new examples — waiting for a full chunk."
                    )

            # ── STOP training (user returned) ────────────────────────────────
            elif idle_sec < 5.0 and training_proc is not None:
                logger.warning(f"User activity detected (idle={idle_sec:.1f}s) — aborting training")
                training_proc.terminate()
                try:
                    training_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    training_proc.kill()
                training_proc = None
                logger.info("Training aborted. VRAM freed. Watching...")

            # ── Training finished on its own ─────────────────────────────────
            elif training_proc is not None and training_proc.poll() is not None:
                rc = training_proc.returncode
                logger.info(f"Training process finished (exit code {rc}). Watching...")
                training_proc = None

            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Idle monitor stopped by user.")
        if training_proc:
            training_proc.terminate()


if __name__ == "__main__":
    start_monitor(idle_threshold=300)
