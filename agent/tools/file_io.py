# =============================================================
# agent/tools/file_io.py — File reading, writing, and listing
#
# TOOLS PROVIDED:
#   read_file      — Read the contents of any text file
#   write_file     — Write or overwrite a file
#   list_directory — List files in a directory (like 'ls')
#   file_exists    — Check if a file or folder exists
#
# SAFETY:
#   All file operations are sandboxed to allowed_base_path
#   (configured in agent.yaml → tools.file_io.allowed_base_path)
#   The agent CANNOT access files outside this directory.
#   This prevents the AI from accidentally reading system files
#   or writing to dangerous locations.
# =============================================================

import os
from pathlib import Path
from typing import Any, Optional

import structlog

from agent.tools.base import BaseTool, ToolResult
from agent.config import get_config

logger = structlog.get_logger(__name__)

# Maximum file size we'll read — prevents the agent from accidentally
# trying to read a 2GB binary file and running out of memory
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _resolve_safe_path(path_str: str, base_path: Path) -> Optional[Path]:
    """
    Resolve a path and verify it's inside the allowed base directory.

    This is the security function. It prevents "path traversal attacks"
    where someone might try to escape the sandbox with paths like:
      "../../etc/passwd"    ← tries to go up two levels
      "/absolute/path"      ← tries to use absolute path

    'resolve()' converts any path to its absolute, canonical form.
    Then we check if the result starts with base_path.
    If not — access denied.

    Returns the resolved Path if safe, None if not allowed.
    """
    try:
        # If path_str is relative (e.g. "src/main.py"),
        # resolve it relative to base_path
        if not Path(path_str).is_absolute():
            full_path = (base_path / path_str).resolve()
        else:
            full_path = Path(path_str).resolve()

        # Check the resolved path is inside the allowed base
        # '.is_relative_to()' returns True if full_path is inside base_path
        base_resolved = base_path.resolve()
        if not str(full_path).startswith(str(base_resolved)):
            logger.warning(
                "path_traversal_blocked",
                requested_path=path_str,
                resolved_path=str(full_path),
                allowed_base=str(base_resolved),
            )
            return None

        return full_path
    except Exception:
        return None


# =============================================================
# READ FILE TOOL
# =============================================================

class ReadFileTool(BaseTool):
    """
    Reads the contents of a text file and returns it as a string.
    The AI uses this to examine source code, configs, logs, etc.
    """

    def __init__(self):
        # Get the allowed base path from config
        config = get_config()
        self._base_path = Path(
            config.tools.file_io.allowed_base_path
        ).resolve()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the complete contents of a text file and return it as a string. "
            "Use this to examine source code, configuration files, logs, README files, "
            "test files, or any other text-based file. "
            "Supports all text encodings. Maximum file size: 10MB. "
            "Do NOT use for binary files (images, executables, zip files)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file, relative to the project root. "
                        "Examples: 'main.py', 'agent/core.py', 'tests/test_core.py'"
                    )
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding. Default is 'utf-8'.",
                    "default": "utf-8"
                }
            },
            "required": ["path"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Step 1: Validate required arguments
        error = self._validate_required_args(arguments, ["path"])
        if error:
            return ToolResult.fail(error)

        path_str = arguments["path"]
        encoding = arguments.get("encoding", "utf-8")

        # Step 2: Security check — is this path inside our sandbox?
        safe_path = _resolve_safe_path(path_str, self._base_path)
        if safe_path is None:
            return ToolResult.fail(
                f"Access denied: '{path_str}' is outside the allowed directory. "
                f"Only files within the project directory can be read."
            )

        # Step 3: Check the file exists
        if not safe_path.exists():
            return ToolResult.fail(
                f"File not found: '{path_str}'. "
                f"Use list_directory to see what files exist."
            )

        if not safe_path.is_file():
            return ToolResult.fail(
                f"'{path_str}' is a directory, not a file. "
                f"Use list_directory to see its contents."
            )

        # Step 4: Check file size before reading
        file_size = safe_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            return ToolResult.fail(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(limit is 10MB). Consider reading specific sections."
            )

        # Step 5: Read the file
        try:
            content = safe_path.read_text(encoding=encoding)

            logger.info(
                "file_read",
                path=path_str,
                size_bytes=file_size,
                lines=content.count('\n') + 1,
            )

            # Return both the content AND useful metadata
            return ToolResult.ok(
                output=content,
                metadata={
                    "path": str(safe_path),
                    "size_bytes": file_size,
                    "lines": content.count('\n') + 1,
                    "encoding": encoding,
                }
            )

        except UnicodeDecodeError:
            return ToolResult.fail(
                f"Cannot read '{path_str}' as text with encoding '{encoding}'. "
                f"This may be a binary file. Try encoding='latin-1' for legacy files."
            )
        except PermissionError:
            return ToolResult.fail(
                f"Permission denied reading '{path_str}'."
            )
        except Exception as e:
            return ToolResult.fail(f"Unexpected error reading file: {e}")


# =============================================================
# WRITE FILE TOOL
# =============================================================

class WriteFileTool(BaseTool):
    """
    Writes content to a file, creating it if it doesn't exist.
    Creates parent directories automatically if needed.
    """

    def __init__(self):
        config = get_config()
        self._base_path = Path(
            config.tools.file_io.allowed_base_path
        ).resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write text content to a file. Creates the file if it doesn't exist. "
            "Overwrites the file if it already exists (the previous content is lost). "
            "Automatically creates any missing parent directories. "
            "Use this to create new source files, update configuration files, "
            "write test files, or save generated code. "
            "For small edits to existing files, prefer providing the full updated content."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write, relative to project root."
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to write to the file."
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding. Default is 'utf-8'.",
                    "default": "utf-8"
                }
            },
            "required": ["path", "content"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["path", "content"])
        if error:
            return ToolResult.fail(error)

        path_str = arguments["path"]
        content = arguments["content"]
        encoding = arguments.get("encoding", "utf-8")

        # Security check
        safe_path = _resolve_safe_path(path_str, self._base_path)
        if safe_path is None:
            return ToolResult.fail(
                f"Access denied: '{path_str}' is outside the allowed directory."
            )

        try:
            # Create parent directories if they don't exist
            # parents=True → create all intermediate dirs
            # exist_ok=True → don't error if they already exist
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            safe_path.write_text(content, encoding=encoding)

            lines = content.count('\n') + 1
            size = len(content.encode(encoding))

            logger.info(
                "file_written",
                path=path_str,
                size_bytes=size,
                lines=lines,
            )

            return ToolResult.ok(
                output=f"Successfully wrote {lines} lines ({size} bytes) to '{path_str}'.",
                metadata={"path": str(safe_path), "size_bytes": size, "lines": lines}
            )

        except PermissionError:
            return ToolResult.fail(f"Permission denied writing to '{path_str}'.")
        except Exception as e:
            return ToolResult.fail(f"Unexpected error writing file: {e}")


# =============================================================
# LIST DIRECTORY TOOL
# =============================================================

class ListDirectoryTool(BaseTool):
    """
    Lists the contents of a directory — files and subdirectories.
    Like running 'ls -la' but with richer output for the AI.
    """

    def __init__(self):
        config = get_config()
        self._base_path = Path(
            config.tools.file_io.allowed_base_path
        ).resolve()

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Returns files and subdirectories "
            "with their sizes and types. Use this to explore the project structure, "
            "find files, or understand how a codebase is organised. "
            "Use '.' for the project root directory."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path relative to project root. "
                        "Use '.' for the root. Examples: '.', 'agent', 'agent/tools'"
                    ),
                    "default": "."
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.'). Default false.",
                    "default": False
                }
            },
            "required": []
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        path_str = arguments.get("path", ".")
        show_hidden = arguments.get("show_hidden", False)

        safe_path = _resolve_safe_path(path_str, self._base_path)
        if safe_path is None:
            return ToolResult.fail(
                f"Access denied: '{path_str}' is outside the allowed directory."
            )

        if not safe_path.exists():
            return ToolResult.fail(f"Directory not found: '{path_str}'")

        if not safe_path.is_dir():
            return ToolResult.fail(
                f"'{path_str}' is a file, not a directory. Use read_file to read it."
            )

        try:
            entries = []

            # Sort: directories first, then files, both alphabetically
            items = sorted(
                safe_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )

            for item in items:
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith('.'):
                    continue

                if item.is_dir():
                    # Count items inside the directory
                    try:
                        child_count = len(list(item.iterdir()))
                        entries.append(f"📁 {item.name}/  ({child_count} items)")
                    except PermissionError:
                        entries.append(f"📁 {item.name}/  (no access)")
                else:
                    # Show file size in a human-readable format
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / 1024 / 1024:.1f}MB"
                    entries.append(f"📄 {item.name}  ({size_str})")

            if not entries:
                output = f"Directory '{path_str}' is empty."
            else:
                output = f"Contents of '{path_str}' ({len(entries)} items):\n\n"
                output += "\n".join(entries)

            return ToolResult.ok(output=output, metadata={"path": str(safe_path)})

        except PermissionError:
            return ToolResult.fail(f"Permission denied reading directory '{path_str}'.")
        except Exception as e:
            return ToolResult.fail(f"Unexpected error listing directory: {e}")


# =============================================================
# FILE EXISTS TOOL
# =============================================================

class FileExistsTool(BaseTool):
    """Quick check: does this file or directory exist?"""

    def __init__(self):
        config = get_config()
        self._base_path = Path(
            config.tools.file_io.allowed_base_path
        ).resolve()

    @property
    def name(self) -> str:
        return "file_exists"

    @property
    def description(self) -> str:
        return (
            "Check whether a file or directory exists at the given path. "
            "Returns true/false with the type (file or directory). "
            "Use before reading or writing to avoid unnecessary errors."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to check, relative to project root."
                }
            },
            "required": ["path"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["path"])
        if error:
            return ToolResult.fail(error)

        path_str = arguments["path"]
        safe_path = _resolve_safe_path(path_str, self._base_path)

        if safe_path is None:
            return ToolResult.fail(
                f"Access denied: '{path_str}' is outside the allowed directory."
            )

        exists = safe_path.exists()
        if exists:
            kind = "directory" if safe_path.is_dir() else "file"
            return ToolResult.ok(
                f"'{path_str}' EXISTS — it is a {kind}.",
                metadata={"exists": True, "type": kind}
            )
        else:
            return ToolResult.ok(
                f"'{path_str}' does NOT exist.",
                metadata={"exists": False}
            )
