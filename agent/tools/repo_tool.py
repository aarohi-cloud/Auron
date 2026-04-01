# =============================================================
# agent/tools/repo_tool.py — Repository intelligence as agent tools
#
# TOOLS PROVIDED:
#   scan_repository    — Scan the whole codebase, build the map
#   find_in_repo       — Find files by concept/name/class/function
#   repo_summary       — Get a high-level overview of the codebase
#   what_imports       — What does this file import?
#   what_depends_on    — What other files import this file?
#
# HOW IT WORKS:
#   These tools wrap the RepoScanner. When the AI needs to understand
#   the codebase, it calls scan_repository first, then uses the
#   other tools to query the resulting map.
# =============================================================

from pathlib import Path
from typing import Any, Optional

import structlog

from agent.tools.base import BaseTool, ToolResult
from agent.repo.scanner import RepoScanner, RepoMap
from agent.config import get_config

logger = structlog.get_logger(__name__)

# Module-level cache: we only scan once per session unless forced
_repo_map_cache: Optional[RepoMap] = None


def _get_repo_map(force: bool = False) -> RepoMap:
    """Get the current repo map, scanning if necessary."""
    global _repo_map_cache
    if _repo_map_cache is None or force:
        config = get_config()
        root = Path(config.tools.file_io.allowed_base_path).resolve()
        scanner = RepoScanner(str(root))
        _repo_map_cache = scanner.scan(force=force)
    return _repo_map_cache


class ScanRepositoryTool(BaseTool):
    """Scan the entire codebase and build a complete understanding map."""

    @property
    def name(self) -> str:
        return "scan_repository"

    @property
    def description(self) -> str:
        return (
            "Scan the entire project codebase and build a complete map of: "
            "all files, imports, classes, functions, test coverage, "
            "circular imports, and dead code. "
            "Run this FIRST whenever you need to understand an unfamiliar codebase "
            "or answer questions about the project structure. "
            "Results are cached — subsequent calls are fast unless force=true. "
            "After scanning, use find_in_repo, what_depends_on, and repo_summary "
            "to query the results."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force a fresh scan, ignoring the cache. Default false.",
                    "default": False
                }
            },
            "required": []
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        force = arguments.get("force", False)

        try:
            repo_map = _get_repo_map(force=force)

            output_lines = [
                f"Repository scan complete.",
                f"",
                repo_map.summary(),
                f"",
                f"Use these tools to explore the results:",
                f"  repo_summary       — High-level overview",
                f"  find_in_repo       — Find files by name/concept/class/function",
                f"  what_imports       — See what a file imports",
                f"  what_depends_on    — See what imports a given file",
            ]

            return ToolResult.ok(
                output="\n".join(output_lines),
                metadata={
                    "total_files": repo_map.total_files,
                    "total_lines": repo_map.total_lines,
                    "untested_count": len(repo_map.untested_files),
                    "circular_imports": len(repo_map.circular_imports),
                }
            )
        except Exception as e:
            return ToolResult.fail(f"Repository scan failed: {e}")


class FindInRepoTool(BaseTool):
    """Find files in the repository by concept, name, class, or function."""

    @property
    def name(self) -> str:
        return "find_in_repo"

    @property
    def description(self) -> str:
        return (
            "Search the repository for files related to a concept, class name, "
            "function name, or any keyword. "
            "Use this to answer: 'Where is X defined?', 'Which files handle Y?', "
            "'Where is the authentication logic?' without needing to know file paths. "
            "Run scan_repository first if you haven't already."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for. Can be a class name, function name, "
                        "concept, or keyword. "
                        "Examples: 'Agent', 'payment', 'authentication', 'config', 'test'"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return. Default 10.",
                    "default": 10
                }
            },
            "required": ["query"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["query"])
        if error:
            return ToolResult.fail(error)

        query = arguments["query"].strip()
        max_results = int(arguments.get("max_results", 10))

        try:
            repo_map = _get_repo_map()
            results = repo_map.find_file(query)[:max_results]

            if not results:
                return ToolResult.ok(
                    f"No files found matching '{query}'.\n"
                    f"Try a different search term, or run scan_repository if you haven't yet."
                )

            lines = [f"Files matching '{query}' ({len(results)} found):\n"]

            for fi in results:
                lines.append(f"📄 {fi.path}")
                lines.append(f"   Language: {fi.language} | Lines: {fi.line_count} | "
                             f"{'🧪 test file' if fi.is_test else ''}")

                if fi.classes:
                    lines.append(f"   Classes:   {', '.join(fi.classes[:5])}")
                if fi.functions:
                    funcs = [f for f in fi.functions if not f.startswith('_')][:5]
                    if funcs:
                        lines.append(f"   Functions: {', '.join(funcs)}")
                if fi.description:
                    lines.append(f"   Purpose:   {fi.description}")
                lines.append("")

            return ToolResult.ok("\n".join(lines))

        except Exception as e:
            return ToolResult.fail(
                f"Search failed: {e}\nTry running scan_repository first."
            )


class RepoSummaryTool(BaseTool):
    """Get a high-level overview of the repository structure."""

    @property
    def name(self) -> str:
        return "repo_summary"

    @property
    def description(self) -> str:
        return (
            "Get a high-level summary of the entire repository: "
            "languages, file counts, lines of code, test coverage gaps, "
            "circular imports, and unreferenced files. "
            "Use this for a quick overview before diving into specific files. "
            "Run scan_repository first."
        )

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            repo_map = _get_repo_map()

            # Build a detailed summary
            lines = [repo_map.summary(), ""]

            # List all files grouped by directory
            dirs: dict[str, list[str]] = {}
            for path in sorted(repo_map.files.keys()):
                parent = str(Path(path).parent)
                dirs.setdefault(parent, []).append(Path(path).name)

            lines.append("File structure:")
            for dir_path in sorted(dirs.keys()):
                files = dirs[dir_path]
                indent = "  " * (dir_path.count("/") + 1 if dir_path != "." else 0)
                lines.append(f"{indent}📁 {dir_path}/")
                for fname in files[:10]:
                    lines.append(f"{indent}  📄 {fname}")
                if len(files) > 10:
                    lines.append(f"{indent}  ... and {len(files) - 10} more files")

            return ToolResult.ok("\n".join(lines))

        except Exception as e:
            return ToolResult.fail(
                f"Failed to get summary: {e}\nTry running scan_repository first."
            )


class WhatImportsTool(BaseTool):
    """Show what modules/files a given file imports."""

    @property
    def name(self) -> str:
        return "what_imports"

    @property
    def description(self) -> str:
        return (
            "Show what modules and files a given file imports. "
            "Use this to understand a file's dependencies "
            "and what would need to change if you modify it."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file. Example: 'agent/core.py'"
                }
            },
            "required": ["file_path"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["file_path"])
        if error:
            return ToolResult.fail(error)

        file_path = arguments["file_path"].strip()

        try:
            repo_map = _get_repo_map()
            info = repo_map.files.get(file_path)

            if not info:
                # Try partial match
                matches = [p for p in repo_map.files if file_path in p]
                if matches:
                    file_path = matches[0]
                    info = repo_map.files[file_path]
                else:
                    return ToolResult.fail(
                        f"File '{file_path}' not found in repo map. "
                        f"Run scan_repository first, or check the path."
                    )

            imports = info.imports
            if not imports:
                return ToolResult.ok(f"'{file_path}' has no imports.")

            # Separate internal imports (our code) from external (libraries)
            all_paths = set(repo_map.files.keys())
            internal = []
            external = []

            for imp in imports:
                candidate = imp.replace(".", "/") + ".py"
                if candidate in all_paths:
                    internal.append(f"  → {candidate}  (internal)")
                else:
                    external.append(f"  → {imp}  (library)")

            lines = [f"'{file_path}' imports {len(imports)} modules:\n"]
            if internal:
                lines.append("Internal (project files):")
                lines.extend(internal)
            if external:
                lines.append("\nExternal (libraries):")
                lines.extend(external[:10])
                if len(external) > 10:
                    lines.append(f"  ... and {len(external) - 10} more")

            return ToolResult.ok("\n".join(lines))

        except Exception as e:
            return ToolResult.fail(f"Failed: {e}")


class WhatDependsOnTool(BaseTool):
    """Show which files import a given file."""

    @property
    def name(self) -> str:
        return "what_depends_on"

    @property
    def description(self) -> str:
        return (
            "Show which files in the project import a given file. "
            "Use this to understand the impact of changing a file — "
            "'what would break if I modify agent/config.py?' "
            "Run scan_repository first."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file. Example: 'agent/config.py'"
                }
            },
            "required": ["file_path"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["file_path"])
        if error:
            return ToolResult.fail(error)

        file_path = arguments["file_path"].strip()

        try:
            repo_map = _get_repo_map()

            # Try exact match first, then partial
            if file_path not in repo_map.files:
                matches = [p for p in repo_map.files if file_path in p]
                if matches:
                    file_path = matches[0]
                else:
                    return ToolResult.fail(
                        f"File '{file_path}' not found. Run scan_repository first."
                    )

            dependents = repo_map.what_depends_on(file_path)

            if not dependents:
                return ToolResult.ok(
                    f"Nothing in this project imports '{file_path}'.\n"
                    f"It may be an entry point, dead code, or not yet scanned."
                )

            lines = [
                f"{len(dependents)} file(s) depend on '{file_path}':",
                f"(Changing '{file_path}' could affect these files)\n",
            ]
            for dep in sorted(dependents):
                lines.append(f"  ← {dep}")

            return ToolResult.ok("\n".join(lines))

        except Exception as e:
            return ToolResult.fail(f"Failed: {e}")
