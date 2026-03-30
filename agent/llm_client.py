# =============================================================
# agent/llm_client.py — The ONLY file that talks to AI models
#
# DESIGN PRINCIPLE — "Abstraction Layer":
#   Every other file in this project calls:
#       response = await client.complete(messages, tools)
#   They never know or care if that's Claude, Ollama, or Groq.
#   This file handles ALL the differences between providers.
#
# SUPPORTED PROVIDERS:
#   - "anthropic"         → Claude (cloud API)
#   - "ollama"            → Any local model (Llama, Mistral, etc.)
#   - "openai_compatible" → Groq, LM Studio, LocalAI, etc.
#
# TO ADD A NEW PROVIDER IN THE FUTURE:
#   Add a new elif block in the complete() method. That's it.
# =============================================================

# 'asyncio' is Python's library for async (concurrent) programming.
# 'async' lets our agent do multiple things without freezing.
# Example: while waiting for the AI to respond, it can update the UI.
import asyncio

# 'time' lets us track how long API calls take (for logging)
import time

# 'dataclasses' gives us a cleaner way to define simple data containers
from dataclasses import dataclass, field

# 'Any', 'Optional' — type hints that make code easier to understand
from typing import Any, Optional

# Structlog gives us beautiful structured (JSON) log output
import structlog

# Our config system — we read model settings from here
from agent.config import get_config, ModelConfig

# Get a logger for this specific module.
# The logger automatically adds the module name to every log line,
# so you can always see WHERE a log message came from.
logger = structlog.get_logger(__name__)


# =============================================================
# DATA CLASSES — simple containers for structured data
# @dataclass auto-generates __init__, __repr__, etc.
# =============================================================

@dataclass
class Message:
    """
    Represents a single message in a conversation.

    AI models work with conversations — a list of messages
    alternating between "user" (you) and "assistant" (the AI).

    role:    "system" | "user" | "assistant"
    content: The text of the message

    "system" messages are special instructions given to the AI
    at the start, before any conversation. Like a briefing.
    """
    role: str        # "system", "user", or "assistant"
    content: str     # The actual text


@dataclass
class ToolDefinition:
    """
    Describes a tool the AI can choose to use.

    When we send messages to the AI, we also tell it:
    "By the way, you have access to these tools."
    The AI then decides whether to use one, and if so, which one.

    name:        Tool identifier (e.g. "run_shell_command")
    description: Plain English explanation of what the tool does
    parameters:  JSON Schema describing the tool's inputs
    """
    name: str
    description: str
    # 'field(default_factory=dict)' means the default is an empty dict {}
    # We can't write parameters: dict = {} directly in dataclasses
    # because mutable defaults cause bugs in Python
    parameters: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    The structured response we always get back from complete().

    No matter which AI provider we use, the response always
    comes back in this same shape. This is the key benefit
    of the abstraction layer — uniform output.

    content:       The AI's text response
    tool_calls:    If the AI wants to use a tool, details are here
    input_tokens:  How many tokens were in our request (for cost tracking)
    output_tokens: How many tokens were in the response (for cost tracking)
    model:         Which model actually generated this response
    duration_ms:   How long the API call took (for performance tracking)
    """
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    duration_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Convenience: total tokens = input + output."""
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Convenience: True if the AI wants to use a tool."""
        return len(self.tool_calls) > 0


# =============================================================
# THE LLM CLIENT
# =============================================================

class LLMClient:
    """
    Universal LLM client — talks to any AI provider.

    This class has ONE public method: complete()
    That method sends messages to the AI and returns an LLMResponse.

    All the provider-specific logic is hidden in private methods
    (prefixed with underscore) that you never need to call directly.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialise the client.

        If no config is provided, it reads from the global config.
        This lets us create a client easily: client = LLMClient()
        """
        # Use provided config, or fall back to reading agent.yaml
        self.config = config or get_config().model

        # We'll create the actual API client lazily (only when first needed)
        # This avoids import errors if a library isn't installed
        self._client: Any = None

        logger.info(
            "llm_client_initialised",
            provider=self.config.provider,
            model=self.config.name,
        )

    def _get_client(self) -> Any:
        """
        Get (or create) the underlying API client.

        We use "lazy initialisation" — we only create the client
        when it's first needed, not when LLMClient() is constructed.
        This means if you're just running tests that don't call the API,
        you won't get errors about missing API keys.
        """
        # If we already created the client, return the cached version
        if self._client is not None:
            return self._client

        # Create the right client based on which provider is configured
        if self.config.provider == "anthropic":
            # Import here (not at top of file) so that if the 'anthropic'
            # package isn't installed, only code that uses Anthropic fails,
            # not the whole module
            import anthropic

            api_key = self.config.api_key
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Add it to your .env file."
                )
            # AsyncAnthropic is the async version — works with 'await'
            self._client = anthropic.AsyncAnthropic(api_key=api_key)

        elif self.config.provider == "ollama":
            # Ollama uses an OpenAI-compatible API on your local machine.
            # We use the 'openai' library but point it at localhost.
            # This is why we can treat Ollama like any OpenAI-compatible API.
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for Ollama support.\n"
                    "Run: pip install openai"
                )
            self._client = AsyncOpenAI(
                base_url=f"{self.config.ollama_base_url}/v1",
                # Ollama doesn't need a real key, but the library requires one
                api_key="ollama",
            )

        elif self.config.provider == "openai_compatible":
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAI-compatible APIs.\n"
                    "Run: pip install openai"
                )
            self._client = AsyncOpenAI(
                base_url=self.config.openai_compatible_base_url or None,
                api_key=self.config.api_key or "no-key",
            )

        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return self._client

    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send messages to the AI and get a response.

        This is the ONE method all other code uses.
        It handles all providers transparently.

        Args:
            messages:      The conversation history so far
            tools:         Optional list of tools the AI can use
            system_prompt: Optional instructions for the AI's behaviour
                          (e.g. "You are an expert Python developer...")

        Returns:
            LLMResponse: The AI's response, always in the same shape
        """
        # Record start time so we can measure how long the call takes
        start_time = time.monotonic()

        logger.debug(
            "llm_call_starting",
            provider=self.config.provider,
            model=self.config.name,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        # Route to the right provider-specific method
        if self.config.provider == "anthropic":
            response = await self._complete_anthropic(messages, tools, system_prompt)
        elif self.config.provider in ("ollama", "openai_compatible"):
            response = await self._complete_openai_compatible(messages, tools, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        # Calculate how long the API call took
        duration_ms = (time.monotonic() - start_time) * 1000
        response.duration_ms = duration_ms

        logger.info(
            "llm_call_complete",
            provider=self.config.provider,
            model=self.config.name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=round(duration_ms, 1),
        )

        return response

    # ----------------------------------------------------------
    # PRIVATE METHODS — provider-specific implementations
    # You don't call these directly — complete() calls them
    # ----------------------------------------------------------

    async def _complete_anthropic(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]],
        system_prompt: Optional[str],
    ) -> LLMResponse:
        """Send a request to the Anthropic (Claude) API."""
        client = self._get_client()

        # Convert our Message objects to the format Anthropic expects.
        # Anthropic wants: [{"role": "user", "content": "hello"}]
        # We filter out "system" messages — Anthropic takes the system
        # prompt as a separate parameter, not as a message
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"  # system messages handled separately
        ]

        # Build the base API call parameters
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": anthropic_messages,
        }

        # Add system prompt if provided
        if system_prompt:
            kwargs["system"] = system_prompt

        # Convert our ToolDefinition objects to Anthropic's format
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]

        # Make the actual API call.
        # 'await' pauses here until the API responds, but doesn't
        # block other things from happening in the meantime.
        api_response = await client.messages.create(**kwargs)

        # Extract the text response
        # The response can have multiple content blocks, we join them
        content_parts = []
        tool_calls = []

        for block in api_response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                # The AI wants to use a tool!
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=api_response.usage.input_tokens,
            output_tokens=api_response.usage.output_tokens,
            model=api_response.model,
        )

    async def _complete_openai_compatible(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]],
        system_prompt: Optional[str],
    ) -> LLMResponse:
        """
        Send a request to any OpenAI-compatible API.
        This covers: Ollama, Groq, LM Studio, LocalAI, etc.
        """
        client = self._get_client()

        # Build message list — OpenAI format includes system as a message
        openai_messages = []

        # Add system prompt as a system-role message (OpenAI style)
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        # Add conversation messages
        for msg in messages:
            if msg.role != "system":  # avoid duplicating system messages
                openai_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": openai_messages,
        }

        # Convert tools to OpenAI function-calling format
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        api_response = await client.chat.completions.create(**kwargs)

        # Extract response
        choice = api_response.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        # Check if the AI wants to use a tool
        if choice.message.tool_calls:
            import json
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    # Arguments come as a JSON string — parse it to a dict
                    "arguments": json.loads(tc.function.arguments or "{}"),
                })

        # Token usage (some local APIs don't report this — default to 0)
        usage = api_response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.name,
        )
