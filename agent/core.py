# =============================================================
# agent/core.py — The Agent class (Phase 2: with tools + memory)
#
# WHAT'S NEW IN PHASE 2:
#   - Tools are registered and sent to the AI
#   - The agentic loop: AI can call tools, get results, think again
#   - Memory is loaded at session start and queried per message
#   - Human-in-the-loop confirmation for destructive actions
#
# THE AGENTIC LOOP (the heart of how Copilot works):
#   1. User sends message
#   2. Agent sends message + tool list to AI
#   3. AI responds with text OR a tool call request
#   4. If tool call → agent runs the tool → sends result back to AI
#   5. AI reads result and responds again (may call more tools)
#   6. Loop repeats until AI responds with text only
#   7. That final text is shown to the user
# =============================================================

import asyncio
import time
from typing import Optional
from dataclasses import dataclass, field

import structlog

from agent.config import get_config, AgentConfig
from agent.llm_client import LLMClient, Message, LLMResponse, ToolDefinition
from agent.tools.base import BaseTool, ToolResult
from agent.tools.file_io import ReadFileTool, WriteFileTool, ListDirectoryTool, FileExistsTool
from agent.tools.shell import ShellTool
from agent.tools.web_search import WebSearchTool, ReadURLTool
from agent.memory.store import MemoryStore

logger = structlog.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are an expert AI agent specialising in software development, \
debugging, testing, and automation. You have deep knowledge of Python, JavaScript, TypeScript, \
Go, Rust, Java, and many other languages and frameworks.

You have access to powerful tools:
- read_file / write_file / list_directory / file_exists: Work with files in the project
- run_shell_command: Execute terminal commands (run tests, git, pip, etc.)
- web_search / read_url: Search the web and read documentation

Your core principles:
1. ACCURACY: Verify facts using your tools. Read actual files before modifying them.
2. SAFETY: Before doing anything destructive, explain what you're about to do.
3. TRANSPARENCY: Show your reasoning. Explain what tools you're using and why.
4. QUALITY: Write production-grade code — with error handling, type hints, and comments.
5. HONESTY: If a tool fails, say so clearly. Do not make up results.

Workflow for code tasks:
- First use list_directory and read_file to understand the existing code
- Make changes that are consistent with the existing style
- Use run_shell_command to run tests and verify your changes work"""


@dataclass
class SessionStats:
    """Tracks usage statistics for the current session."""
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
    The core AI Agent with tools and memory.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        project: str = "default",
    ):
        self.config = config or get_config()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_client = LLMClient(config=self.config.model)
        self.conversation_history: list[Message] = []
        self.stats = SessionStats()
        self.memory = MemoryStore(project=project)
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

        logger.info(
            "agent_created",
            name=self.config.agent.name,
            provider=self.config.model.provider,
            model=self.config.model.name,
            tools=list(self._tools.keys()),
            project=project,
        )

    def _register_default_tools(self) -> None:
        tools_config = self.config.tools
        if tools_config.file_io.enabled:
            for tool in [ReadFileTool(), WriteFileTool(), ListDirectoryTool(), FileExistsTool()]:
                self.register_tool(tool)
        if tools_config.shell.enabled:
            self.register_tool(ShellTool())
        if tools_config.web_search.enabled:
            self.register_tool(WebSearchTool())
        if tools_config.web_browse.enabled:
            self.register_tool(ReadURLTool())

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self._tools.values()
        ]

    async def chat(self, user_message: str) -> LLMResponse:
        if not user_message or not user_message.strip():
            raise ValueError("Message cannot be empty.")

        self._check_rate_limits()
        self._check_token_budget()

        user_msg = Message(role="user", content=user_message.strip())
        self.conversation_history.append(user_msg)

        logger.debug(
            "chat_message_received",
            message_preview=user_message[:100],
            history_length=len(self.conversation_history),
        )

        system_prompt = self._build_system_prompt(user_message)

        try:
            response = await self._agentic_loop(system_prompt)
        except Exception as e:
            self.conversation_history.pop()
            logger.error("agentic_loop_failed", error=str(e))
            raise

        self.conversation_history.append(
            Message(role="assistant", content=response.content)
        )

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

    async def _agentic_loop(self, system_prompt: str) -> LLMResponse:
        """
        The core agentic loop — keeps calling tools until the AI is done.
        """
        max_iterations = self.config.agent.max_planning_depth
        iteration = 0
        loop_messages = self.conversation_history.copy()
        last_response = None

        while iteration < max_iterations:
            iteration += 1

            logger.debug("agentic_loop_iteration", iteration=iteration)

            response = await self.llm_client.complete(
                messages=loop_messages,
                tools=self._get_tool_definitions(),
                system_prompt=system_prompt,
            )

            last_response = response
            self.stats.total_input_tokens += response.input_tokens
            self.stats.total_output_tokens += response.output_tokens
            self.stats.total_llm_calls += 1

            # No tool calls = AI is done, return the response
            if not response.has_tool_calls:
                logger.debug("agentic_loop_complete", iterations=iteration)
                return response

            # AI responded with text before calling a tool
            if response.content:
                loop_messages.append(
                    Message(role="assistant", content=response.content)
                )

            # Execute all requested tool calls
            for tool_call in response.tool_calls:
                tool_result = await self._execute_tool_call(tool_call)
                result_message = (
                    f"Tool result for '{tool_call['name']}':\n"
                    f"{tool_result.to_string()}"
                )
                loop_messages.append(
                    Message(role="user", content=result_message)
                )
                self.stats.total_tool_calls += 1

        logger.warning("agentic_loop_max_iterations_reached", max_iterations=max_iterations)
        return last_response

    async def _execute_tool_call(self, tool_call: dict) -> ToolResult:
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        logger.info("tool_call_requested", tool=tool_name, arguments_preview=str(arguments)[:100])

        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                f"Tool '{tool_name}' is not available. "
                f"Available: {', '.join(self._tools.keys())}"
            )

        if self._requires_confirmation(tool_name, arguments):
            confirmed = await self._request_confirmation(tool_name, arguments)
            if not confirmed:
                return ToolResult.fail(f"User declined to run '{tool_name}'.")

        try:
            result = await tool.run(arguments)
            logger.info("tool_call_complete", tool=tool_name, success=result.success)
            return result
        except Exception as e:
            logger.error("tool_call_exception", tool=tool_name, error=str(e))
            return ToolResult.fail(f"Tool '{tool_name}' raised an exception: {e}")

    def _requires_confirmation(self, tool_name: str, arguments: dict) -> bool:
        safety = self.config.safety
        if not safety.require_confirmation_for_destructive_actions:
            return False
        if tool_name in safety.auto_approve_actions:
            return False
        if tool_name == "write_file":
            return True
        if tool_name == "run_shell_command":
            command = arguments.get("command", "").lower()
            destructive = ["rm ", "del ", "rmdir", "drop ", "delete ",
                          "truncate", "format", "sudo", "git reset --hard"]
            if any(p in command for p in destructive):
                return True
        return False

    async def _request_confirmation(self, tool_name: str, arguments: dict) -> bool:
        print(f"\n⚠️  Agent wants to run: {tool_name}")
        print(f"   Arguments: {arguments}")
        print(f"   Proceed? [y/N] ", end="", flush=True)
        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, input)
            return answer.strip().lower() in ("y", "yes")
        except Exception:
            return False

    def _build_system_prompt(self, user_message: str) -> str:
        memories_text = self.memory.format_for_prompt(user_message)
        if memories_text:
            return f"{self.system_prompt}\n\n{memories_text}"
        return self.system_prompt

    def remember(self, content: str, category: str = "fact") -> None:
        self.memory.save(content, category=category)

    def clear_history(self) -> None:
        self.conversation_history = []
        logger.info("conversation_history_cleared")

    def get_history(self) -> list[Message]:
        return self.conversation_history.copy()

    def get_stats(self) -> SessionStats:
        return self.stats

    def _check_rate_limits(self) -> None:
        limits = self.config.rate_limits
        if (limits.max_llm_calls_per_minute > 0 and
                self.stats.total_llm_calls >= limits.max_llm_calls_per_minute * 10):
            raise RuntimeError("Rate limit exceeded.")

    def _check_token_budget(self) -> None:
        limits = self.config.rate_limits
        if limits.max_tokens_per_session > 0:
            if self.stats.total_tokens >= limits.max_tokens_per_session:
                raise RuntimeError(
                    f"Token budget exceeded: {self.stats.total_tokens} tokens used."
                )
