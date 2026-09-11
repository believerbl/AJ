"""
AJ Idle Monitor - Phase 7 Watchdog

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
import sys
import ctypes
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import config

# ---------------------------------------------------------------------------
# Logging - idle monitor logs to logs/idle_monitor.log
# Trainer subprocess writes directly to logs/training_process.log
# ---------------------------------------------------------------------------

LOG_DIR = config.BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_LOG = LOG_DIR / "training_process.log"
IDLE_LOG = LOG_DIR / "idle_monitor.log"

logging.basicConfig(
    filename=str(IDLE_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("IdleMonitor")

# Print key events to terminal so you can see status in real time
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
        return max(0.0, elapsed_ms / 1000.0)
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
    """Read the training cursor - how many examples have already been consumed."""
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
    new = max(0, total - cursor)
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
    print(f"[*] Training log -> {TRAINING_LOG}", flush=True)
    print(f"[*] Monitor log  -> {IDLE_LOG}", flush=True)

    training_proc: subprocess.Popen | None = None
    last_failure_time: float = 0.0
    COOLDOWN_ON_ERROR: int = 300  # wait 5 minutes before retrying if trainer crashes
    was_idle: bool = False

    try:
        while True:
            idle_sec = get_idle_seconds()
            ready, new_count = new_examples_available()

            # Track whether system has been idle for at least 60 seconds
            if idle_sec >= 60.0:
                was_idle = True

            # ---------------------------------------------------------------
            # 1. START training (user idle >= threshold and no training active)
            # ---------------------------------------------------------------
            if idle_sec >= idle_threshold and training_proc is None:
                # Check cooldown if previous run failed
                elapsed_since_err = time.time() - last_failure_time
                if elapsed_since_err < COOLDOWN_ON_ERROR:
                    time.sleep(2)
                    continue

                if ready:
                    logger.info(
                        f"Idle {idle_sec:.0f}s | {new_count} new examples "
                        f"(>= chunk size {config.TRAINING_CHUNK_SIZE}) -> launching trainer"
                    )
                    with open(TRAINING_LOG, "a", encoding="utf-8") as lf:
                        lf.write(f"\n{'='*55}\n")
                        lf.write(f"TRAINING RUN - {datetime.now().isoformat()}\n")
                        lf.write(f"New examples this chunk: {new_count}\n")
                        lf.write(f"{'='*55}\n")
                    training_proc = subprocess.Popen(
                        [sys.executable, "-m", "core.train_lora"],
                        stdout=open(TRAINING_LOG, "a", encoding="utf-8"),
                        stderr=subprocess.STDOUT,
                    )
                else:
                    logger.info(
                        f"Idle {idle_sec:.0f}s but only {new_count}/{config.TRAINING_CHUNK_SIZE} "
                        "new examples - waiting for a full chunk."
                    )

            # ---------------------------------------------------------------
            # 2. USER PRESENCE DETECTED (idle_sec < 5.0)
            # ---------------------------------------------------------------
            elif idle_sec < 5.0:
                # Case A: Training is actively running -> abort immediately & free VRAM
                if training_proc is not None:
                    logger.warning(f"User activity detected (idle={idle_sec:.1f}s) - aborting training to free VRAM")
                    training_proc.terminate()
                    try:
                        training_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        training_proc.kill()
                    training_proc = None
                    logger.info("Training aborted. VRAM freed. Watching...")
                    was_idle = False

                # Case B: System was idle, user just returned (no trainer running)
                elif was_idle:
                    logger.info(f"User presence detected (idle reset to {idle_sec:.1f}s). System is active.")
                    was_idle = False

            # ---------------------------------------------------------------
            # 3. Training completed or exited on its own
            # ---------------------------------------------------------------
            if training_proc is not None and training_proc.poll() is not None:
                rc = training_proc.returncode
                if rc == 0:
                    logger.info("Training chunk completed successfully (exit code 0). VRAM freed. Watching...")
                else:
                    logger.error(
                        f"Trainer exited with code {rc}. See {TRAINING_LOG} for full details. "
                        f"Pausing trainer for {COOLDOWN_ON_ERROR}s to avoid busy-loop."
                    )
                    # Extract the real traceback from TRAINING_LOG and print to terminal
                    try:
                        with open(TRAINING_LOG, "r", encoding="utf-8", errors="ignore") as lf:
                            err_lines = [l.rstrip() for l in lf.readlines() if l.strip()]
                            # Grab up to the last 8 lines of the actual error
                            tb_slice = err_lines[-8:] if len(err_lines) >= 8 else err_lines
                            if tb_slice:
                                print("\n[!] --- Trainer Error Output ---", flush=True)
                                for line in tb_slice:
                                    print(f"    {line}")
                                print("[!] ----------------------------\n", flush=True)
                    except Exception:
                        pass
                    last_failure_time = time.time()
                training_proc = None

            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Idle monitor stopped by user.")
        if training_proc:
            training_proc.terminate()


if __name__ == "__main__":
    start_monitor(idle_threshold=300)
