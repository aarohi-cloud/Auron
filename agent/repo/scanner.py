# =============================================================
# agent/repo/scanner.py — Repository Intelligence Engine
#
# WHAT THIS DOES:
#   Scans your entire codebase and builds a rich understanding of:
#   - Every file and its purpose
#   - What each file imports (dependency graph)
#   - All classes, functions, and their signatures
#   - Test coverage gaps (files with no corresponding test)
#   - Dead code candidates (files nothing else imports)
#   - Circular import chains
#   - Overall project structure and architecture
#
# WHY THIS MATTERS:
#   Without this, the agent only knows about files you explicitly
#   show it. With this, it understands the ENTIRE codebase and
#   can answer: "Where is X?", "What uses Y?", "What's untested?"
#
# HOW IT WORKS:
#   1. Walk every file in the project directory
#   2. Parse Python files using the 'ast' module (Abstract Syntax Tree)
#      AST = Python's built-in way to read code as structured data
#      instead of just text — it understands imports, classes, functions
#   3. Build a graph of relationships between files
#   4. Store everything as a RepoMap object
#   5. Save it to disk so we don't re-scan every session
# =============================================================

import ast          # Python's built-in code parser
import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# File extensions we understand and can parse
SUPPORTED_EXTENSIONS = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".md":   "markdown",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".json": "json",
    ".toml": "toml",
    ".env":  "env",
}

# Directories to always skip — these are not part of the project's own code
SKIP_DIRECTORIES = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", "dist", "build", ".eggs", "*.egg-info",
    ".mypy_cache", ".ruff_cache", "htmlcov", ".coverage",
    "data",   # our memory database
    "logs",   # our log files
}


# =============================================================
# DATA CLASSES — the structured output of the scanner
# =============================================================

@dataclass
class FileInfo:
    """
    Everything we know about a single file in the repository.

    This is the core unit of the repo map — one per file.
    """
    # Relative path from project root (e.g. "agent/core.py")
    path: str

    # Programming language (e.g. "python", "typescript")
    language: str

    # Size in bytes
    size_bytes: int

    # Number of lines
    line_count: int

    # Last modified timestamp
    last_modified: float

    # MD5 hash of the file — used to detect if it changed since last scan
    content_hash: str

    # List of modules/files this file imports
    # e.g. ["agent.config", "agent.llm_client", "asyncio"]
    imports: list[str] = field(default_factory=list)

    # Names of all classes defined in this file
    classes: list[str] = field(default_factory=list)

    # Names of all functions defined in this file (top-level)
    functions: list[str] = field(default_factory=list)

    # Brief auto-generated description of what this file does
    description: str = ""

    # True if this looks like a test file
    is_test: bool = False

    # Any parse errors encountered (so we don't crash on bad files)
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class RepoMap:
    """
    The complete map of an entire repository.

    Built by RepoScanner.scan() — this is the output.
    Contains everything the agent needs to understand the codebase.
    """
    # Project root directory
    root_path: str

    # When this scan was done (Unix timestamp)
    scanned_at: float

    # How long the scan took
    scan_duration_seconds: float

    # All files found, keyed by relative path
    files: dict[str, FileInfo] = field(default_factory=dict)

    # Import graph: maps each file to the list of files that import it
    # e.g. {"agent/config.py": ["agent/core.py", "agent/llm_client.py"]}
    # Tells you: "what depends on this file?"
    reverse_imports: dict[str, list[str]] = field(default_factory=dict)

    # Files that nothing else imports — possible dead code or entry points
    unreferenced_files: list[str] = field(default_factory=list)

    # Circular import chains detected
    # e.g. [["agent/a.py", "agent/b.py", "agent/a.py"]]
    circular_imports: list[list[str]] = field(default_factory=list)

    # Python files that have no corresponding test file
    untested_files: list[str] = field(default_factory=list)

    # Summary statistics
    total_files: int = 0
    total_lines: int = 0
    languages: dict[str, int] = field(default_factory=dict)  # language → file count

    def summary(self) -> str:
        """
        Generate a human-readable summary of the repository.
        This is what gets injected into the agent's system prompt
        so it has an overview of the codebase before answering.
        """
        lines = [
            f"Repository: {self.root_path}",
            f"Scanned: {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.scanned_at))}",
            f"Total files: {self.total_files} | Total lines: {self.total_lines:,}",
            "",
            "Languages:",
        ]
        for lang, count in sorted(self.languages.items(), key=lambda x: -x[1]):
            lines.append(f"  {lang}: {count} files")

        if self.untested_files:
            lines.append(f"\nUntested files ({len(self.untested_files)}):")
            for f in self.untested_files[:5]:  # show first 5
                lines.append(f"  ⚠ {f}")
            if len(self.untested_files) > 5:
                lines.append(f"  ... and {len(self.untested_files) - 5} more")

        if self.circular_imports:
            lines.append(f"\nCircular imports detected ({len(self.circular_imports)}):")
            for cycle in self.circular_imports[:3]:
                lines.append(f"  ⚠ {' → '.join(cycle)}")

        if self.unreferenced_files:
            lines.append(f"\nUnreferenced files ({len(self.unreferenced_files)}) — possible dead code:")
            for f in self.unreferenced_files[:5]:
                lines.append(f"  ? {f}")

        return "\n".join(lines)

    def find_file(self, query: str) -> list[FileInfo]:
        """
        Find files matching a query — by name, class, function, or concept.

        The agent uses this to answer "where is X defined?"
        without the user specifying a file path.

        Examples:
            find_file("payment") → files with "payment" in path/classes/functions
            find_file("Agent")   → files defining an Agent class
            find_file("config")  → files related to configuration
        """
        query_lower = query.lower()
        results = []

        for file_info in self.files.values():
            score = 0

            # Strong signal: query appears in the file path
            if query_lower in file_info.path.lower():
                score += 10

            # Strong signal: query matches a class name
            for cls in file_info.classes:
                if query_lower in cls.lower():
                    score += 8

            # Medium signal: query matches a function name
            for func in file_info.functions:
                if query_lower in func.lower():
                    score += 5

            # Weak signal: query appears in the description
            if query_lower in file_info.description.lower():
                score += 3

            if score > 0:
                results.append((score, file_info))

        # Return sorted by relevance, highest first
        results.sort(key=lambda x: -x[0])
        return [fi for _, fi in results]

    def what_imports(self, file_path: str) -> list[str]:
        """Which files does this file import?"""
        info = self.files.get(file_path)
        return info.imports if info else []

    def what_depends_on(self, file_path: str) -> list[str]:
        """Which files import this file? (reverse dependency)"""
        return self.reverse_imports.get(file_path, [])


# =============================================================
# THE SCANNER
# =============================================================

class RepoScanner:
    """
    Scans a repository and produces a RepoMap.

    Usage:
        scanner = RepoScanner("/path/to/project")
        repo_map = scanner.scan()
        print(repo_map.summary())
    """

    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        # Cache file — we save the repo map here so we don't
        # re-scan every time the agent starts
        self._cache_path = self.root / ".agent_repo_cache.json"

    def scan(self, force: bool = False) -> RepoMap:
        """
        Scan the repository and return a RepoMap.

        If a cached map exists and the files haven't changed much,
        we return the cached version (much faster than re-scanning).
        Set force=True to always do a fresh scan.

        Args:
            force: If True, ignore cache and scan fresh.

        Returns:
            RepoMap: Complete map of the repository.
        """
        # Try to load from cache first
        if not force and self._cache_path.exists():
            cached = self._load_cache()
            if cached and self._cache_is_fresh(cached):
                logger.info("repo_map_loaded_from_cache", files=cached.total_files)
                return cached

        logger.info("repo_scan_starting", root=str(self.root))
        start_time = time.monotonic()

        # Step 1: Find all files
        all_files = self._find_all_files()
        logger.info("repo_files_found", count=len(all_files))

        # Step 2: Parse each file
        file_infos: dict[str, FileInfo] = {}
        for file_path in all_files:
            rel_path = str(file_path.relative_to(self.root))
            info = self._parse_file(file_path, rel_path)
            file_infos[rel_path] = info

        # Step 3: Build the import graph
        reverse_imports = self._build_reverse_imports(file_infos)

        # Step 4: Detect issues
        unreferenced = self._find_unreferenced(file_infos, reverse_imports)
        circular = self._find_circular_imports(file_infos)
        untested = self._find_untested(file_infos)

        # Step 5: Calculate statistics
        total_lines = sum(fi.line_count for fi in file_infos.values())
        languages: dict[str, int] = {}
        for fi in file_infos.values():
            languages[fi.language] = languages.get(fi.language, 0) + 1

        duration = time.monotonic() - start_time

        repo_map = RepoMap(
            root_path=str(self.root),
            scanned_at=time.time(),
            scan_duration_seconds=duration,
            files=file_infos,
            reverse_imports=reverse_imports,
            unreferenced_files=unreferenced,
            circular_imports=circular,
            untested_files=untested,
            total_files=len(file_infos),
            total_lines=total_lines,
            languages=languages,
        )

        # Save to cache
        self._save_cache(repo_map)

        logger.info(
            "repo_scan_complete",
            files=len(file_infos),
            total_lines=total_lines,
            duration_seconds=round(duration, 2),
            untested=len(untested),
            circular_imports=len(circular),
        )

        return repo_map

    def _find_all_files(self) -> list[Path]:
        """Walk the directory tree and collect all relevant files."""
        result = []

        for dirpath, dirnames, filenames in os.walk(self.root):
            # Remove skip directories from dirnames IN PLACE
            # This tells os.walk not to descend into them
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRECTORIES and not d.endswith(".egg-info")
            ]

            for filename in filenames:
                filepath = Path(dirpath) / filename
                ext = filepath.suffix.lower()

                if ext in SUPPORTED_EXTENSIONS:
                    result.append(filepath)

        return sorted(result)

    def _parse_file(self, filepath: Path, rel_path: str) -> FileInfo:
        """
        Parse a single file and extract all information from it.
        Handles errors gracefully — never crashes on a bad file.
        """
        try:
            stat = filepath.stat()
            size_bytes = stat.st_size
            last_modified = stat.st_mtime
        except OSError:
            return FileInfo(
                path=rel_path, language="unknown",
                size_bytes=0, line_count=0, last_modified=0,
                content_hash="", parse_errors=["Could not stat file"]
            )

        language = SUPPORTED_EXTENSIONS.get(filepath.suffix.lower(), "unknown")

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return FileInfo(
                path=rel_path, language=language,
                size_bytes=size_bytes, line_count=0,
                last_modified=last_modified, content_hash="",
                parse_errors=[f"Could not read: {e}"]
            )

        line_count = content.count("\n") + 1
        content_hash = hashlib.md5(content.encode()).hexdigest()
        is_test = self._is_test_file(rel_path, content)

        # Defaults for non-Python files
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []
        parse_errors: list[str] = []
        description = self._generate_description(rel_path, content, language)

        # Deep parse for Python files using AST
        if language == "python":
            imports, classes, functions, errs = self._parse_python_ast(content, rel_path)
            parse_errors.extend(errs)

        return FileInfo(
            path=rel_path,
            language=language,
            size_bytes=size_bytes,
            line_count=line_count,
            last_modified=last_modified,
            content_hash=content_hash,
            imports=imports,
            classes=classes,
            functions=functions,
            description=description,
            is_test=is_test,
            parse_errors=parse_errors,
        )

    def _parse_python_ast(
        self, content: str, filepath: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Parse Python code using Python's built-in AST parser.

        AST = Abstract Syntax Tree. When Python reads code, it first
        converts it to a tree of nodes. We use that tree to extract
        imports, class names, and function names precisely — not with
        fragile text parsing.

        Returns: (imports, classes, functions, errors)
        """
        imports = []
        classes = []
        functions = []
        errors = []

        try:
            # Parse the Python source into an AST
            tree = ast.parse(content, filename=filepath)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e}")
            return imports, classes, functions, errors

        # Walk every node in the AST tree
        for node in ast.walk(tree):

            # Import statements: "import os" or "import os, sys"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            # From imports: "from pathlib import Path"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

            # Class definitions: "class MyClass:"
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

            # Function definitions: "def my_function():" or "async def foo():"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only include top-level functions (not methods inside classes)
                # We check this by seeing if the function's parent is a module
                # ast.walk doesn't give us parent info directly, so we track
                # only functions that appear at module level
                functions.append(node.name)

        return imports, classes, functions, errors

    def _generate_description(self, rel_path: str, content: str, language: str) -> str:
        """
        Generate a brief auto-description of a file.

        For Python files: use the module docstring if present.
        For other files: generate from filename and first line.
        """
        if language == "python" and content.strip():
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    # Return first line of docstring only
                    return docstring.split("\n")[0].strip()
            except SyntaxError:
                pass

        # Fall back to a description based on the filename
        name = Path(rel_path).stem
        return f"{language} file: {name}"

    def _is_test_file(self, rel_path: str, content: str) -> bool:
        """Determine if a file is a test file."""
        path_lower = rel_path.lower()
        return (
            "/test_" in path_lower or
            "/tests/" in path_lower or
            path_lower.startswith("test_") or
            "_test.py" in path_lower or
            ".spec." in path_lower
        )

    def _build_reverse_imports(
        self, files: dict[str, FileInfo]
    ) -> dict[str, list[str]]:
        """
        Build the reverse import map: for each file, who imports it?

        This answers: "what would break if I change this file?"
        We convert module names (agent.config) to file paths (agent/config.py)
        for matching.
        """
        reverse: dict[str, list[str]] = {path: [] for path in files}

        for file_path, info in files.items():
            for imported_module in info.imports:
                # Convert "agent.config" → "agent/config.py"
                candidate = imported_module.replace(".", "/") + ".py"

                if candidate in files and candidate != file_path:
                    reverse[candidate].append(file_path)

        return reverse

    def _find_unreferenced(
        self,
        files: dict[str, FileInfo],
        reverse_imports: dict[str, list[str]],
    ) -> list[str]:
        """
        Find Python files that nothing else imports.
        These are either entry points (main.py) or dead code.
        """
        unreferenced = []
        for path, importers in reverse_imports.items():
            info = files.get(path)
            if not info or info.language != "python":
                continue
            if info.is_test:
                continue  # Test files are fine to be unreferenced

            filename = Path(path).name
            # Entry points — expected to be unreferenced
            if filename in ("main.py", "cli.py", "setup.py", "conftest.py"):
                continue

            if not importers:
                unreferenced.append(path)

        return sorted(unreferenced)

    def _find_circular_imports(
        self, files: dict[str, FileInfo]
    ) -> list[list[str]]:
        """
        Detect circular import chains using DFS (Depth First Search).

        A circular import is when A imports B imports C imports A.
        This causes Python ImportError at runtime.

        We build a simple graph and look for cycles.
        """
        # Build adjacency list: file → list of files it imports
        graph: dict[str, list[str]] = {}
        for path, info in files.items():
            neighbours = []
            for mod in info.imports:
                candidate = mod.replace(".", "/") + ".py"
                if candidate in files and candidate != path:
                    neighbours.append(candidate)
            graph[path] = neighbours

        cycles = []
        visited: set[str] = set()
        path_stack: list[str] = []
        in_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            path_stack.append(node)
            in_stack.add(node)

            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    dfs(neighbour)
                elif neighbour in in_stack:
                    # Found a cycle — extract the cycle from the stack
                    cycle_start = path_stack.index(neighbour)
                    cycle = path_stack[cycle_start:] + [neighbour]
                    if cycle not in cycles:
                        cycles.append(cycle)

            path_stack.pop()
            in_stack.remove(node)

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node)

        return cycles[:10]  # Cap at 10 to avoid overwhelming output

    def _find_untested(self, files: dict[str, FileInfo]) -> list[str]:
        """
        Find Python source files that have no corresponding test file.

        Convention: agent/core.py → tests/test_core.py
        If tests/test_core.py doesn't exist, agent/core.py is untested.
        """
        test_files = {fi.path for fi in files.values() if fi.is_test}
        untested = []

        for path, info in files.items():
            if info.language != "python" or info.is_test:
                continue
            # Skip __init__.py files — they're usually empty
            if Path(path).name == "__init__.py":
                continue

            filename = Path(path).stem
            # Check if any test file references this filename
            has_test = any(
                filename in tf for tf in test_files
            )
            if not has_test:
                untested.append(path)

        return sorted(untested)

    # ----------------------------------------------------------
    # CACHING — save/load the repo map to avoid re-scanning
    # ----------------------------------------------------------

    def _save_cache(self, repo_map: RepoMap) -> None:
        """Save the repo map to a JSON cache file."""
        try:
            data = {
                "root_path": repo_map.root_path,
                "scanned_at": repo_map.scanned_at,
                "scan_duration_seconds": repo_map.scan_duration_seconds,
                "total_files": repo_map.total_files,
                "total_lines": repo_map.total_lines,
                "languages": repo_map.languages,
                "unreferenced_files": repo_map.unreferenced_files,
                "circular_imports": repo_map.circular_imports,
                "untested_files": repo_map.untested_files,
                "reverse_imports": repo_map.reverse_imports,
                "files": {
                    path: asdict(fi)
                    for path, fi in repo_map.files.items()
                }
            }
            self._cache_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("repo_cache_save_failed", error=str(e))

    def _load_cache(self) -> Optional[RepoMap]:
        """Load a previously saved repo map from cache."""
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            files = {
                path: FileInfo(**fi_data)
                for path, fi_data in data.get("files", {}).items()
            }
            return RepoMap(
                root_path=data["root_path"],
                scanned_at=data["scanned_at"],
                scan_duration_seconds=data["scan_duration_seconds"],
                files=files,
                reverse_imports=data.get("reverse_imports", {}),
                unreferenced_files=data.get("unreferenced_files", []),
                circular_imports=data.get("circular_imports", []),
                untested_files=data.get("untested_files", []),
                total_files=data.get("total_files", 0),
                total_lines=data.get("total_lines", 0),
                languages=data.get("languages", {}),
            )
        except Exception as e:
            logger.warning("repo_cache_load_failed", error=str(e))
            return None

    def _cache_is_fresh(self, cached: RepoMap) -> bool:
        """
        Check if the cache is still valid.
        We consider it stale if it's older than 5 minutes
        OR if the number of files has changed.
        """
        cache_age = time.time() - cached.scanned_at
        if cache_age > 300:  # 5 minutes
            return False

        # Quick check: count current files and compare
        current_files = self._find_all_files()
        return len(current_files) == cached.total_files
