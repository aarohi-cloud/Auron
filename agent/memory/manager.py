# =============================================================
# agent/memory/manager.py — Decides what to remember and when
#
# DIFFERENCE FROM store.py:
#   store.py   = the database (save, search, delete raw memories)
#   manager.py = the intelligence layer on top:
#                - Detects if a user message contains something
#                  worth saving automatically
#                - Formats memories for injection into prompts
#                - Provides the CLI-facing commands (save, list, delete)
#                - Deduplicates — avoids saving the same thing twice
#
# MEMORY CATEGORIES:
#   preference  — How the user likes things done
#                 "I prefer tabs over spaces"
#                 "Always add type hints to Python functions"
#   fact        — True things about the project or environment
#                 "This project uses PostgreSQL 15"
#                 "The API runs on port 8080"
#   decision    — Architectural or design choices made
#                 "We decided to use FastAPI over Django"
#                 "Authentication uses JWT, not sessions"
#   pattern     — Recurring patterns or conventions
#                 "All service classes go in agent/services/"
#                 "Error messages always include an error code"
#   error       — Known bugs or issues to watch for
#                 "ChromaDB crashes if the path has spaces"
# =============================================================

import re
from typing import Optional

import structlog

from agent.memory.store import MemoryStore, Memory
from agent.config import get_config

logger = structlog.get_logger(__name__)

# Keywords that suggest a message contains something worth remembering.
# If a user message contains any of these patterns, we flag it
# as a candidate for saving.
_MEMORY_TRIGGER_PATTERNS = [
    # Explicit save requests
    r"\bremember\b",
    r"\bdon'?t forget\b",
    r"\bkeep in mind\b",
    r"\bnote that\b",
    r"\bfyi\b",
    # Preference statements
    r"\bi (always|never|prefer|like|hate|use|don'?t use)\b",
    r"\bwe (always|never|prefer|use|decided)\b",
    r"\bour (convention|standard|rule|policy|practice)\b",
    # Project facts
    r"\bthis project (uses|runs|requires|depends)\b",
    r"\bthe (api|server|database|backend|frontend) (runs|uses|is)\b",
    # Decision markers
    r"\bwe decided\b",
    r"\bwe chose\b",
    r"\bgoing forward\b",
    r"\bfrom now on\b",
]

# Compile patterns once for efficiency (not every call)
_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _MEMORY_TRIGGER_PATTERNS
]


class MemoryManager:
    """
    High-level memory management — the intelligence layer over MemoryStore.

    This is what the Agent and CLI interact with.
    MemoryStore is used internally by this class.

    Usage:
        manager = MemoryManager(project="my-project")

        # Save something explicitly
        manager.save("User prefers snake_case for variables", "preference")

        # Check if a message contains something worth auto-saving
        if manager.should_auto_save(user_message):
            manager.save(user_message, "fact")

        # Get memories to inject into a prompt
        context = manager.build_context_for(user_message)
    """

    def __init__(self, project: Optional[str] = None):
        config = get_config()
        # Use the project from config if not explicitly provided
        self._project = project or config.memory.project
        self._store = MemoryStore(project=self._project)
        self._similarity_threshold = config.memory.similarity_threshold

        logger.info(
            "memory_manager_created",
            project=self._project,
            backend=config.memory.backend,
        )

    # ----------------------------------------------------------
    # SAVING MEMORIES
    # ----------------------------------------------------------

    def save(
        self,
        content: str,
        category: str = "fact",
        metadata: Optional[dict] = None,
    ) -> Memory:
        """
        Save a memory explicitly.

        Called when the user types: remember: I use tabs not spaces

        Args:
            content:  What to remember
            category: One of: preference, fact, decision, pattern, error
            metadata: Any extra context (source file, timestamp, etc.)

        Returns:
            The saved Memory object
        """
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        # Validate category
        valid_categories = {"preference", "fact", "decision", "pattern", "error"}
        if category not in valid_categories:
            category = "fact"  # default to fact if unknown category given

        # Check for near-duplicates before saving
        # This prevents saving "I use pytest" five times
        existing = self._store.search(content, top_k=3)
        for mem in existing:
            if self._is_duplicate(content, mem.content):
                logger.info(
                    "memory_duplicate_skipped",
                    new_content=content[:60],
                    existing_content=mem.content[:60],
                )
                # Return the existing memory rather than saving a duplicate
                return mem

        memory = self._store.save(content, category=category, metadata=metadata or {})
        logger.info("memory_saved_explicit", category=category, content_preview=content[:80])
        return memory

    def should_auto_save(self, message: str) -> bool:
        """
        Check whether a user message contains something worth auto-saving.

        We scan for trigger patterns (preference statements, decisions, etc.)
        Returns True if the message looks like it contains memorable information.

        The agent calls this after each user message. If True, it offers
        to save the memory or saves it automatically depending on config.

        Example triggers:
            "I always use black for Python formatting" → True
            "We decided to use PostgreSQL" → True
            "What is a decorator?" → False
            "Run my tests" → False
        """
        message_lower = message.lower().strip()

        # Too short to be meaningful
        if len(message_lower) < 10:
            return False

        # Check against our trigger patterns
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(message_lower):
                return True

        return False

    def extract_memory_content(self, message: str) -> tuple[str, str]:
        """
        Extract the memory content and guess its category from a message.

        When auto-saving, we clean up the message a bit before storing it
        and try to determine what category it belongs to.

        Returns:
            Tuple of (cleaned_content, category)

        Examples:
            "remember that I use tabs" → ("I use tabs", "preference")
            "fyi this project uses FastAPI" → ("This project uses FastAPI", "fact")
            "we decided to use JWT" → ("We decided to use JWT", "decision")
        """
        content = message.strip()

        # Remove explicit save prefixes so we store the clean fact.
        # Order matters — longer/more specific patterns come first.
        # We stop after the FIRST match to avoid double-stripping.
        #
        # Examples of what gets cleaned:
        #   "remember that I use tabs"   → "I use tabs"
        #   "remember: I use tabs"       → "I use tabs"
        #   "remember I use tabs"        → "I use tabs"
        #   "don't forget I love coffee" → "I love coffee"
        #   "fyi this uses FastAPI"      → "this uses FastAPI"
        prefixes_to_strip = [
            r"^remember\s+that\s+",          # "remember that ..."
            r"^remember\s*:?\s*",            # "remember:" or "remember "
            r"^don'?t\s+forget\s+that\s+", # "don't forget that ..."
            r"^don'?t\s+forget\s*:?\s*",   # "don't forget:"
            r"^note\s+that\s+",              # "note that ..."
            r"^note\s*:?\s*",                # "note:"
            r"^fyi\s*:?\s*",                 # "fyi:"
            r"^keep\s+in\s+mind\s+that\s+",
            r"^keep\s+in\s+mind\s*:?\s*",
            r"^just\s+so\s+you\s+know\s*:?\s*",
        ]
        for prefix in prefixes_to_strip:
            stripped = re.sub(prefix, "", content, flags=re.IGNORECASE).strip()
            if stripped != content:
                # A prefix matched — use the cleaned version and stop
                content = stripped
                break

        # Guess the category from the cleaned content
        category = self._guess_category(content)

        # Capitalise first letter for consistency in stored memories
        if content:
            content = content[0].upper() + content[1:]

        return content, category

    def _guess_category(self, content: str) -> str:
        """Guess what category a piece of content belongs to."""
        lower = content.lower()

        if any(w in lower for w in ["prefer", "always", "never", "like", "hate", "use", "don't use"]):
            return "preference"
        if any(w in lower for w in ["decided", "chose", "going forward", "from now on"]):
            return "decision"
        if any(w in lower for w in ["convention", "pattern", "standard", "rule"]):
            return "pattern"
        if any(w in lower for w in ["bug", "crash", "error", "issue", "broken", "fails"]):
            return "error"
        return "fact"

    def _is_duplicate(self, new_content: str, existing_content: str) -> bool:
        """
        Check if two memory strings are essentially the same thing.

        We do a simple word-overlap check — if 80%+ of the words in the
        new content already appear in an existing memory, it's a duplicate.
        Not perfect, but good enough to prevent obvious repeated saves.
        """
        new_words = set(new_content.lower().split())
        existing_words = set(existing_content.lower().split())

        if not new_words:
            return False

        overlap = len(new_words & existing_words) / len(new_words)
        return overlap > 0.8

    # ----------------------------------------------------------
    # RETRIEVING MEMORIES
    # ----------------------------------------------------------

    def build_context_for(self, query: str) -> str:
        """
        Search for memories relevant to a query and format them
        as text to prepend to the system prompt.

        This is the main method the Agent calls before every LLM request.
        It makes the agent "remember" things automatically.

        Returns empty string if no relevant memories found.

        Example output:
            "Relevant memories from previous sessions:
             - [preference] User always uses Black for Python formatting
             - [fact] This project uses FastAPI on port 8080
             - [decision] We decided to use JWT for authentication"
        """
        memories = self._store.search(query, top_k=5)

        if not memories:
            return ""

        lines = ["Relevant memories from previous sessions:"]
        for mem in memories:
            lines.append(f"- [{mem.category}] {mem.content}")

        return "\n".join(lines)

    def get_all(self) -> list[Memory]:
        """Get all saved memories for the current project."""
        return self._store.get_all()

    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by its ID.
        Returns True if deleted, False if not found.
        """
        return self._store.delete(memory_id)

    def clear_all(self) -> int:
        """Delete ALL memories for this project. Returns count deleted."""
        return self._store.clear_all()

    def format_memories_for_display(self) -> str:
        """
        Format all memories as a readable table for CLI display.

        This is what the 'memories' command shows the user.
        """
        all_memories = self.get_all()

        if not all_memories:
            return f"No memories saved for project '{self._project}' yet.\n\nTip: Say 'remember: <something>' to save a memory."

        # Group by category for readability
        by_category: dict[str, list[Memory]] = {}
        for mem in all_memories:
            by_category.setdefault(mem.category, []).append(mem)

        lines = [f"Memories for project '{self._project}' ({len(all_memories)} total)\n"]

        # Category display order and emoji
        category_display = [
            ("preference", "🎯 Preferences"),
            ("decision",   "🏛  Decisions"),
            ("fact",       "📌 Facts"),
            ("pattern",    "🔄 Patterns"),
            ("error",      "⚠️  Known Issues"),
        ]

        for cat_key, cat_label in category_display:
            mems = by_category.get(cat_key, [])
            if not mems:
                continue
            lines.append(f"{cat_label}:")
            for mem in mems:
                # Show first 8 chars of ID so user can reference it for deletion
                short_id = mem.memory_id[:8]
                lines.append(f"  [{short_id}] {mem.content}")
            lines.append("")

        lines.append("To delete: type 'forget <id>' using the first 8 characters shown above.")
        return "\n".join(lines)
