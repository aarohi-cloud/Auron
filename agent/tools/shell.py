# =============================================================
# agent/tools/shell.py — Run terminal/shell commands
#
# WHAT THIS ENABLES:
#   The agent can now run ANY shell command and read its output.
#   Examples:
#     - "Run my tests" → runs: pytest tests/ -v
#     - "Check git status" → runs: git status
#     - "Install this package" → runs: pip install requests
#     - "List Python processes" → runs: ps aux | grep python
#
# SAFETY:
#   - Commands run with a configurable timeout (default 30s)
#   - Commands run in the project directory (not system root)
#   - stdout AND stderr are both captured and returned to the AI
#   - Exit code is reported — the AI knows if a command failed
#
# NOTE ON SECURITY:
#   Shell execution is powerful and inherently risky.
#   In Phase 5 (Safety), we add the human-in-the-loop confirmation
#   gate that asks you before running anything destructive.
#   For now, the agent runs what you ask it to.
# =============================================================

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

import structlog

from agent.tools.base import BaseTool, ToolResult
from agent.config import get_config

logger = structlog.get_logger(__name__)


class ShellTool(BaseTool):
    """
    Executes shell commands and returns their output.

    The agent can run any terminal command — tests, git operations,
    package management, scripts, etc.

    Both stdout (normal output) and stderr (error output) are
    captured and returned so the AI has the full picture.
    """

    def __init__(self):
        config = get_config()
        self._timeout = config.tools.shell.timeout_seconds
        # Commands run from the project root directory
        self._cwd = Path(
            config.tools.file_io.allowed_base_path
        ).resolve()

    @property
    def name(self) -> str:
        return "run_shell_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output (stdout + stderr). "
            "Use this to: run tests (pytest, jest, cargo test), check git status, "
            "install packages (pip install, npm install), run scripts, "
            "inspect processes, check disk usage, compile code, or any other "
            "terminal operation. "
            f"Commands time out after {self._timeout} seconds. "
            "Returns the exit code, stdout, and stderr. "
            "The working directory is the project root."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The shell command to run. "
                        "Examples: 'pytest tests/ -v', 'git status', "
                        "'pip install requests', 'python main.py --help'"
                    )
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Timeout in seconds. Default is {self._timeout}. "
                        "Increase for long-running commands like test suites."
                    ),
                    "default": self._timeout
                }
            },
            "required": ["command"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["command"])
        if error:
            return ToolResult.fail(error)

        command = arguments["command"].strip()
        timeout = int(arguments.get("timeout", self._timeout))

        if not command:
            return ToolResult.fail("Command cannot be empty.")

        logger.info(
            "shell_command_starting",
            command=command,
            timeout=timeout,
            cwd=str(self._cwd),
        )

        try:
            # asyncio.create_subprocess_shell runs the command asynchronously.
            # This means the agent doesn't freeze while the command runs —
            # it can still handle other things.
            #
            # asyncio.subprocess.PIPE means we capture the output
            # instead of printing it directly to the terminal.
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,   # capture stdout
                stderr=asyncio.subprocess.PIPE,   # capture stderr
                cwd=str(self._cwd),               # run in project dir
                # Pass current environment variables to the subprocess
                # This ensures pip, python, git etc. are found correctly
                env=os.environ.copy(),
            )

            # Wait for the process to finish, with a timeout.
            # 'communicate()' reads all stdout and stderr.
            # If timeout expires, asyncio raises TimeoutError.
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process — it's taking too long
                process.kill()
                await process.communicate()  # Clean up

                return ToolResult.fail(
                    f"Command timed out after {timeout} seconds: {command}\n"
                    f"Tip: Increase the timeout parameter for long-running commands."
                )

            # Decode bytes to string
            # 'errors=replace' means garbled characters become '?' instead of crashing
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode

            logger.info(
                "shell_command_complete",
                command=command,
                exit_code=exit_code,
                stdout_lines=stdout.count('\n'),
                stderr_lines=stderr.count('\n'),
            )

            # Build a clear, structured output the AI can easily parse
            output_parts = []

            output_parts.append(f"Command: {command}")
            output_parts.append(f"Exit code: {exit_code} ({'success' if exit_code == 0 else 'FAILED'})")
            output_parts.append(f"Working directory: {self._cwd}")

            if stdout.strip():
                output_parts.append(f"\n--- STDOUT ---\n{stdout.strip()}")
            else:
                output_parts.append("\n--- STDOUT --- (empty)")

            if stderr.strip():
                output_parts.append(f"\n--- STDERR ---\n{stderr.strip()}")

            output = "\n".join(output_parts)

            # success = exit code 0 (Unix convention: 0 = success, anything else = failure)
            success = exit_code == 0

            return ToolResult(
                success=success,
                output=output,
                # If the command failed, put the error info in the error field too
                error=None if success else f"Command failed with exit code {exit_code}",
                metadata={
                    "exit_code": exit_code,
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )

        except FileNotFoundError:
            # This happens when the command itself doesn't exist
            # e.g. running 'pytest' when pytest isn't installed
            cmd_name = command.split()[0]
            return ToolResult.fail(
                f"Command not found: '{cmd_name}'. "
                f"Make sure it's installed and in your PATH. "
                f"Tip: Try 'pip install {cmd_name}' or 'npm install -g {cmd_name}'."
            )
        except PermissionError:
            return ToolResult.fail(
                f"Permission denied running: {command}"
            )
        except Exception as e:
            logger.error("shell_command_error", command=command, error=str(e))
            return ToolResult.fail(f"Unexpected error running command: {e}")
