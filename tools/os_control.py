import subprocess
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OSControlTool:
    """
    Tool for safe local execution of Windows scripts and commands.
    """
    def __init__(self, work_dir: str = "."):
        self.work_dir = os.path.abspath(work_dir)

    def execute_command(self, command: str, timeout_sec: int = 30) -> str:
        """
        Execute a shell command with a timeout.
        """
        logger.info(f"Executing OS command: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
                
            return output.strip() if output.strip() else "Command executed successfully with no output."
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout_sec} seconds: {command}")
            return f"Error: Command timed out after {timeout_sec} seconds."
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return f"Error executing command: {str(e)}"

    def execute(self, params: Dict[str, Any]) -> str:
        """
        Execute wrapper for agentic JSON calling.
        Expects params {"command": "..."}
        """
        command = params.get("command", "")
        if not command:
            return "Error: No command provided."
            
        # Optional: Add safeguards here to reject dangerous commands (e.g., format, rm -rf)
        forbidden = ["format", "del /s", "rmdir /s"]
        for f in forbidden:
            if f in command.lower():
                return f"Error: Command contains forbidden pattern '{f}'."
                
        return self.execute_command(command)
