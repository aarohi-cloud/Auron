# =============================================================
# agent/memory/store.py — Persistent memory across sessions
#
# WHAT THIS DOES:
#   Lets the agent remember things between restarts.
#   "I prefer tabs over spaces" → saved, recalled next session.
#   "This project uses FastAPI" → saved, recalled when relevant.
#
# HOW IT WORKS (Two modes):
#   1. in_memory  — fast, but forgotten when agent stops (Phase 1 default)
#   2. chroma     — saved to disk as a vector database, persists forever
#
# WHAT IS A VECTOR DATABASE?
#   Normal databases search by exact match: "find rows where name='Alice'"
#   A vector database searches by MEANING: "find memories similar to this idea"
#   This means if you saved "I use tabs", searching for "indentation preference"
#   still finds it — because the MEANING is similar, even if the words differ.
#   ChromaDB is a free, local vector database that runs on your machine.
# =============================================================

import json
import time
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

import structlog

from agent.config import get_config

logger = structlog.get_logger(__name__)


@dataclass
class Memory:
    """
    A single memory — one fact or piece of information the agent remembers.

    content:    The actual text being remembered
    category:   Type of memory ('preference', 'fact', 'decision', 'pattern')
    project:    Which project this belongs to (namespace — prevents mixing projects)
    created_at: Unix timestamp when this was saved
    memory_id:  Unique identifier for this memory
    metadata:   Any extra info (source file, conversation turn, etc.)
    """
    content: str
    category: str = "fact"
    project: str = "default"
    created_at: float = field(default_factory=time.time)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    """
    Manages persistent memories across sessions.

    Two backends are supported:
    - InMemoryBackend: Simple dict — fast, no persistence (for testing/Phase 1)
    - ChromaBackend:   Vector database on disk — semantic search, persistent

    The backend is selected from agent.yaml (memory.backend).
    Switching backends doesn't change how you USE MemoryStore —
    same API either way. That's the abstraction pattern again.
    """

    def __init__(self, project: str = "default"):
        """
        Args:
            project: Project namespace. Memories are isolated per project.
                    "my-web-app" memories don't bleed into "my-api" memories.
        """
        config = get_config()
        self._project = project
        self._backend_type = config.memory.backend
        self._top_k = config.memory.top_k_results
        self._backend = self._create_backend(config)

        logger.info(
            "memory_store_created",
            backend=self._backend_type,
            project=project,
        )

    def _create_backend(self, config: Any) -> "_MemoryBackend":
        """Create the right backend based on config."""
        if self._backend_type == "in_memory":
            return _InMemoryBackend()
        elif self._backend_type == "chroma":
            return _ChromaBackend(
                persist_path=config.memory.path,
                project=self._project,
            )
        else:
            raise ValueError(f"Unknown memory backend: {self._backend_type}")

    def save(
        self,
        content: str,
        category: str = "fact",
        metadata: Optional[dict] = None
    ) -> Memory:
        """
        Save a new memory.

        Args:
            content:  What to remember (e.g. "User prefers tabs over spaces")
            category: Type — 'preference', 'fact', 'decision', 'pattern', 'error'
            metadata: Any extra context to store alongside the memory

        Returns:
            The saved Memory object (includes its assigned ID)

        Example:
            store.save("User uses pytest for all Python testing", category="preference")
        """
        memory = Memory(
            content=content,
            category=category,
            project=self._project,
            metadata=metadata or {},
        )
        self._backend.add(memory)

        logger.info(
            "memory_saved",
            memory_id=memory.memory_id,
            category=category,
            content_preview=content[:80],
        )

        return memory

    def search(self, query: str, top_k: Optional[int] = None) -> list[Memory]:
        """
        Find memories relevant to a query.

        In ChromaDB backend: uses semantic (meaning-based) search.
        In InMemory backend: uses simple string matching.

        Args:
            query: What you're looking for (plain English)
            top_k: How many memories to return (default from config)

        Returns:
            List of Memory objects, most relevant first

        Example:
            memories = store.search("code style preferences")
            # Returns "User prefers tabs" even if query words differ
        """
        k = top_k or self._top_k
        results = self._backend.search(query, k)

        logger.debug(
            "memory_search",
            query=query,
            results_found=len(results),
        )

        return results

    def get_all(self) -> list[Memory]:
        """Return ALL memories for the current project."""
        return self._backend.get_all()

    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by its ID.

        Returns True if deleted, False if not found.
        """
        result = self._backend.delete(memory_id)
        if result:
            logger.info("memory_deleted", memory_id=memory_id)
        return result

    def clear_all(self) -> int:
        """
        Delete ALL memories for this project.
        Returns the number of memories deleted.
        """
        count = len(self.get_all())
        self._backend.clear()
        logger.info("memory_cleared", project=self._project, count=count)
        return count

    def format_for_prompt(self, query: str) -> str:
        """
        Search for relevant memories and format them as text
        to inject into the agent's system prompt.

        This is how the agent "remembers" — we search for relevant
        memories and prepend them to the conversation context.

        Returns empty string if no relevant memories found.
        """
        memories = self.search(query)
        if not memories:
            return ""

        lines = ["Relevant memories from previous sessions:"]
        for mem in memories:
            lines.append(f"- [{mem.category}] {mem.content}")

        return "\n".join(lines)


# =============================================================
# BACKEND IMPLEMENTATIONS (internal — not used directly)
# =============================================================

class _MemoryBackend:
    """Abstract interface for memory backends."""

    def add(self, memory: Memory) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[Memory]:
        raise NotImplementedError

    def get_all(self) -> list[Memory]:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class _InMemoryBackend(_MemoryBackend):
    """
    Simple in-memory storage using a Python list.
    Fast, no dependencies, but memories are lost on restart.
    Perfect for development and testing.
    """

    def __init__(self):
        # Just a list of Memory objects
        self._memories: list[Memory] = []

    def add(self, memory: Memory) -> None:
        self._memories.append(memory)

    def search(self, query: str, top_k: int) -> list[Memory]:
        """
        Simple keyword search — checks if any query word
        appears in the memory content (case-insensitive).
        Not semantic, but good enough for Phase 1.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each memory by how many query words it contains
        scored = []
        for mem in self._memories:
            content_lower = mem.content.lower()
            matches = sum(1 for word in query_words if word in content_lower)
            if matches > 0:
                scored.append((matches, mem))

        # Sort by score (highest first) and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def get_all(self) -> list[Memory]:
        return self._memories.copy()

    def delete(self, memory_id: str) -> bool:
        original_count = len(self._memories)
        self._memories = [m for m in self._memories if m.memory_id != memory_id]
        return len(self._memories) < original_count

    def clear(self) -> None:
        self._memories = []


class _ChromaBackend(_MemoryBackend):
    """
    ChromaDB vector database backend.
    Memories persist to disk and are searchable by meaning.
    Requires: pip install chromadb
    """

    def __init__(self, persist_path: str, project: str):
        self._project = project
        self._persist_path = persist_path
        self._collection = None  # Lazy init

    def _get_collection(self):
        """Get or create the ChromaDB collection (lazy initialisation)."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Run: pip install chromadb\n"
                "Or switch to in_memory backend in agent.yaml"
            )

        # Create the ChromaDB client (persists to disk)
        client = chromadb.PersistentClient(path=self._persist_path)

        # Each project gets its own "collection" (isolated namespace)
        # 'get_or_create_collection' is safe to call repeatedly
        safe_name = f"agent_{self._project}".replace("-", "_").replace(" ", "_")
        self._collection = client.get_or_create_collection(
            name=safe_name,
            # Cosine similarity = good for text similarity
            metadata={"hnsw:space": "cosine"}
        )

        return self._collection

    def add(self, memory: Memory) -> None:
        collection = self._get_collection()
        collection.add(
            ids=[memory.memory_id],
            documents=[memory.content],
            # Store full memory data as metadata so we can reconstruct it
            metadatas=[{
                "category": memory.category,
                "project": memory.project,
                "created_at": memory.created_at,
                "metadata_json": json.dumps(memory.metadata),
            }]
        )

    def search(self, query: str, top_k: int) -> list[Memory]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                content = results["documents"][0][i]
                memories.append(Memory(
                    memory_id=mem_id,
                    content=content,
                    category=meta.get("category", "fact"),
                    project=meta.get("project", self._project),
                    created_at=float(meta.get("created_at", 0)),
                    metadata=json.loads(meta.get("metadata_json", "{}")),
                ))

        return memories

    def get_all(self) -> list[Memory]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.get()
        memories = []
        for i, mem_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            content = results["documents"][i]
            memories.append(Memory(
                memory_id=mem_id,
                content=content,
                category=meta.get("category", "fact"),
                project=meta.get("project", self._project),
                created_at=float(meta.get("created_at", 0)),
                metadata=json.loads(meta.get("metadata_json", "{}")),
            ))
        return memories

    def delete(self, memory_id: str) -> bool:
        collection = self._get_collection()
        try:
            collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        collection = self._get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
