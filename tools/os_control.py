import subprocess
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Commands too dangerous to ever run, regardless of context
FORBIDDEN = ["format", "del /s", "rmdir /s", "rm -rf", "shutdown", "rd /s"]


def run_os_control(command: str, timeout: int = 15) -> str:
    """
    Execute a shell command (or a block of Python code) safely.
    Designed to be called by tool_node with the flat schema:
        {"tool": "os_control", "input": "<command or python code>"}

    - If input looks like a Python script (contains newlines or 'import'),
      it is written to a temp .py file and executed with the current venv Python.
    - Otherwise it is treated as a raw shell command.
    - stdout + stderr are always captured and returned as a string.
    - Hard timeout of 15 s prevents runaway processes.
    """
    if not command.strip():
        return "Error: empty command."

    # Safety gate
    for bad in FORBIDDEN:
        if bad in command.lower():
            return f"Error: blocked forbidden pattern '{bad}'."

    is_script = "\n" in command or "import " in command

    try:
        if is_script:
            # Write to a temp file and run with venv Python
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(command)
                tmp_path = f.name

            logger.info(f"[os_control] running python script: {tmp_path}")
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            os.unlink(tmp_path)
        else:
            logger.info(f"[os_control] running shell command: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        return output.strip() or "Command executed successfully with no output."

    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    except Exception as e:
        logger.error(f"[os_control] failed: {e}")
        return f"Error: {e}"
