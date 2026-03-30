# =============================================================
# agent/tools/base.py — The blueprint every tool must follow
#
# CONCEPT — Abstract Base Class:
#   This file defines a "contract" that all tools must honour.
#   Any class that inherits from BaseTool MUST implement:
#     - name       (property) → the tool's identifier
#     - description (property) → plain English explanation
#     - parameters (property) → JSON Schema of inputs
#     - run()      (method)   → actually executes the tool
#
#   If a subclass forgets to implement any of these,
#   Python raises a TypeError immediately — not a mystery
#   crash later. This is "fail fast" design.
#
# CONCEPT — JSON Schema:
#   The AI model needs to know what inputs each tool accepts.
#   We describe inputs using JSON Schema — a standard format
#   for describing the shape of data.
#   Example: {"type": "object", "properties": {"path": {"type": "string"}}}
#   The AI reads this and knows it should pass a "path" string.
# =============================================================

# 'abc' = Abstract Base Classes — Python's built-in module
# for defining abstract classes (blueprints)
from abc import ABC, abstractmethod

# 'Any' = any Python type
# 'Optional' = might be None
# 'Union' = can be one of several types
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# =============================================================
# TOOL RESULT — the standard shape of every tool's output
# =============================================================

@dataclass
class ToolResult:
    """
    The standardised result from running any tool.

    Every tool returns one of these — success or failure.
    This uniform shape means core.py can handle all tool
    results the same way, regardless of which tool ran.

    success:  True if the tool ran without errors
    output:   The tool's output as a string (shown to the AI)
    error:    If success=False, what went wrong
    metadata: Optional extra info (e.g. file size, exit code)
    """
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_string(self) -> str:
        """
        Convert the result to a string the AI can read.

        When a tool finishes, we send its result back to the AI
        as text. This method formats it clearly.
        """
        if self.success:
            return self.output
        else:
            # Clearly mark failures so the AI knows something went wrong
            return f"ERROR: {self.error}\n\nOutput (if any):\n{self.output}"

    @classmethod
    def ok(cls, output: str, metadata: Optional[dict] = None) -> "ToolResult":
        """
        Convenience method to create a success result.

        Instead of: ToolResult(success=True, output="done")
        Write:       ToolResult.ok("done")
        """
        return cls(success=True, output=output, metadata=metadata or {})

    @classmethod
    def fail(cls, error: str, output: str = "") -> "ToolResult":
        """
        Convenience method to create a failure result.

        Instead of: ToolResult(success=False, error="file not found", output="")
        Write:       ToolResult.fail("file not found")
        """
        return cls(success=False, output=output, error=error)


# =============================================================
# BASE TOOL — the abstract blueprint
# =============================================================

class BaseTool(ABC):
    """
    Abstract base class that every tool must inherit from.

    'ABC' means Abstract Base Class — a class that cannot be
    instantiated directly. You can't do: tool = BaseTool()
    You CAN do: tool = ShellTool() (a concrete subclass)

    The @abstractmethod decorator means "subclasses MUST
    implement this — no exceptions."
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The tool's unique identifier.

        This is what the AI uses to call the tool.
        Must be snake_case (e.g. "run_shell_command", "read_file").
        No spaces or special characters.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Plain English explanation of what the tool does.

        The AI reads this to decide WHEN to use the tool.
        Write it as if explaining to a smart person what
        this tool is for. Be specific about capabilities
        and limitations.

        Good:  "Read the contents of a file. Returns the full
                text content. Use for source code, configs,
                logs. Max file size 10MB."
        Bad:   "Reads files."
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        JSON Schema describing the tool's input parameters.

        The AI uses this to know what arguments to provide
        when calling the tool. Follow JSON Schema format.

        Example:
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8"
                }
            },
            "required": ["path"]  # These args are mandatory
        }
        """
        ...

    @abstractmethod
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given arguments.

        This is where the actual work happens.
        'async' because some tools need to wait for
        external operations (network, disk, subprocess).

        Args:
            arguments: Dict of argument names to values.
                      Matches the structure defined in parameters.

        Returns:
            ToolResult: Always returns this — never raises
                       exceptions out of run(). Catch errors
                       inside and return ToolResult.fail(...)
        """
        ...

    def to_tool_definition(self) -> dict:
        """
        Convert this tool to the format the LLM client expects.

        Called by core.py to build the list of tools sent to the AI.
        You don't override this — it's the same for every tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def _validate_required_args(
        self,
        arguments: dict,
        required: list[str]
    ) -> Optional[str]:
        """
        Check that all required arguments are present.

        Returns an error message string if validation fails,
        or None if everything is fine.

        Tools call this at the start of run() to catch
        missing arguments early with a clear error.

        Example usage in a tool's run() method:
            error = self._validate_required_args(arguments, ["path"])
            if error:
                return ToolResult.fail(error)
        """
        missing = [arg for arg in required if arg not in arguments]
        if missing:
            return (
                f"Missing required arguments: {', '.join(missing)}. "
                f"Received: {list(arguments.keys())}"
            )
        return None

    def __repr__(self) -> str:
        """How this tool appears when printed — useful for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}')"
