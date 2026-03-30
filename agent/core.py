# =============================================================
# agent/core.py — The Agent class. The heart of everything.
#
# WHAT THIS DOES:
#   - Holds the conversation history
#   - Has a default system prompt (the agent's "personality")
#   - Sends messages to the LLM via llm_client.py
#   - Returns responses to the caller
#   - Tracks token usage across the session
#   - Enforces rate limits and token budgets
#
# WHAT IT DOES NOT DO (yet — added in later phases):
#   - Execute tools (Phase 2)
#   - Store long-term memories (Phase 2)
#   - Plan complex multi-step tasks (built on top of this, Phase 3)
# =============================================================

import asyncio
import time
from typing import Optional
from dataclasses import dataclass, field

import structlog

from agent.config import get_config, AgentConfig
from agent.llm_client import LLMClient, Message, LLMResponse, ToolDefinition

logger = structlog.get_logger(__name__)

# This is the default system prompt — the agent's "personality" and rules.
# A system prompt is like a briefing you give the AI before any conversation.
# It shapes every response the AI gives.
DEFAULT_SYSTEM_PROMPT = """You are an expert AI agent specialising in software development, \
debugging, testing, and automation. You have deep knowledge of Python, JavaScript, TypeScript, \
Go, Rust, Java, and many other languages and frameworks.

Your core principles:
1. ACCURACY: Never guess at API signatures or library methods. If unsure, say so clearly.
2. SAFETY: Before doing anything destructive (deleting files, running risky commands), \
confirm with the user.
3. TRANSPARENCY: Show your reasoning. Explain what you're doing and why.
4. QUALITY: Write production-grade code — with error handling, type hints, and clear comments.
5. HONESTY: Distinguish between what you know confidently, what you're inferring, \
and what you cannot verify.

When helping with code:
- Understand the full context before proposing changes
- Propose changes that are consistent with the existing codebase style
- Always consider edge cases and error handling
- Suggest tests alongside any code changes"""


@dataclass
class SessionStats:
    """
    Tracks usage statistics for the current session.
    Used for rate limiting, cost tracking, and observability.
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    session_start_time: float = field(default_factory=time.monotonic)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def session_duration_seconds(self) -> float:
        return time.monotonic() - self.session_start_time


class Agent:
    """
    The core AI Agent.

    This is the main class you interact with. Create one instance,
    then call chat() repeatedly to have a conversation.

    The agent maintains the full conversation history so it always
    has context of what was said before — just like a real conversation.

    Basic usage:
        agent = Agent()
        response = await agent.chat("Help me write a Python function to sort a list")
        print(response.content)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Create a new Agent instance.

        Args:
            config:        AgentConfig object. If None, reads from agent.yaml
            system_prompt: Custom instructions for the AI's behaviour.
                          If None, uses the DEFAULT_SYSTEM_PROMPT above.
        """
        # Load config (from agent.yaml + env vars)
        self.config = config or get_config()

        # The system prompt defines the agent's personality and rules
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # The LLM client is our connection to the AI model
        # It's the only part that knows which AI provider we're using
        self.llm_client = LLMClient(config=self.config.model)

        # Conversation history — a list of all messages so far
        # This is what gives the agent "memory" within a session
        # Every time we call chat(), we add to this list and send
        # the WHOLE list to the AI, so it always has full context
        self.conversation_history: list[Message] = []

        # Session statistics — track tokens, calls, etc.
        self.stats = SessionStats()

        # Tools the agent can use (populated in Phase 2)
        self.tools: list[ToolDefinition] = []

        logger.info(
            "agent_created",
            name=self.config.agent.name,
            provider=self.config.model.provider,
            model=self.config.model.name,
        )

    async def chat(self, user_message: str) -> LLMResponse:
        """
        Send a message to the agent and get a response.

        This is the main method you call to interact with the agent.
        It handles the full cycle:
          1. Add user message to history
          2. Check rate limits / token budget
          3. Send full history to the AI
          4. Add AI response to history
          5. Update session statistics
          6. Return the response

        Args:
            user_message: What you want to say to the agent

        Returns:
            LLMResponse: The agent's response (includes text, any tool
                        calls the AI wants to make, and token usage)

        Example:
            response = await agent.chat("What is a decorator in Python?")
            print(response.content)
        """
        # --- Step 1: Validate the input ---
        if not user_message or not user_message.strip():
            raise ValueError("Message cannot be empty.")

        # --- Step 2: Check rate limits ---
        self._check_rate_limits()

        # --- Step 3: Check token budget ---
        self._check_token_budget()

        # --- Step 4: Add user message to conversation history ---
        # This is how the AI "remembers" what you said earlier —
        # we send the whole history every time
        user_msg = Message(role="user", content=user_message.strip())
        self.conversation_history.append(user_msg)

        logger.debug(
            "chat_message_received",
            message_preview=user_message[:100],  # log first 100 chars only
            history_length=len(self.conversation_history),
        )

        # --- Step 5: Send to the AI ---
        try:
            response = await self.llm_client.complete(
                messages=self.conversation_history,
                tools=self.tools if self.tools else None,
                system_prompt=self.system_prompt,
            )
        except Exception as e:
            # If the API call fails, remove the user message we just added
            # so the history stays consistent (no orphaned user messages)
            self.conversation_history.pop()
            logger.error("llm_call_failed", error=str(e))
            raise

        # --- Step 6: Add AI response to conversation history ---
        # This is critical — the AI needs to see its own previous responses
        # to maintain coherent conversation
        assistant_msg = Message(role="assistant", content=response.content)
        self.conversation_history.append(assistant_msg)

        # --- Step 7: Update session statistics ---
        self.stats.total_input_tokens += response.input_tokens
        self.stats.total_output_tokens += response.output_tokens
        self.stats.total_llm_calls += 1

        logger.info(
            "chat_response_generated",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_session_tokens=self.stats.total_tokens,
            has_tool_calls=response.has_tool_calls,
        )

        return response

    def clear_history(self) -> None:
        """
        Clear the conversation history.

        Use this to start a fresh conversation without creating
        a new Agent instance (which would reload config).
        """
        self.conversation_history = []
        logger.info("conversation_history_cleared")

    def get_history(self) -> list[Message]:
        """
        Get the full conversation history.

        Returns a copy (not the original list) so callers can't
        accidentally modify the agent's history.
        """
        return self.conversation_history.copy()

    def get_stats(self) -> SessionStats:
        """Get the current session statistics."""
        return self.stats

    def _check_rate_limits(self) -> None:
        """
        Check if we're within configured rate limits.
        Raises RuntimeError if a limit would be exceeded.

        NOTE: This is a simplified check for Phase 1.
        Phase 7 (Observability) adds proper time-window rate limiting.
        """
        limits = self.config.rate_limits

        # Check LLM call limit (simple count for now)
        # A proper implementation would use a sliding time window
        if (limits.max_llm_calls_per_minute > 0 and
                self.stats.total_llm_calls >= limits.max_llm_calls_per_minute * 10):
            raise RuntimeError(
                f"Rate limit: exceeded {limits.max_llm_calls_per_minute * 10} "
                f"LLM calls this session."
            )

    def _check_token_budget(self) -> None:
        """
        Check if we're within the configured token budget.
        Raises RuntimeError if the budget would be exceeded.
        """
        limits = self.config.rate_limits

        # 0 means unlimited
        if limits.max_tokens_per_session > 0:
            if self.stats.total_tokens >= limits.max_tokens_per_session:
                raise RuntimeError(
                    f"Token budget exceeded: used {self.stats.total_tokens} tokens "
                    f"(limit: {limits.max_tokens_per_session} per session).\n"
                    f"Start a new session or increase max_tokens_per_session in agent.yaml."
                )
