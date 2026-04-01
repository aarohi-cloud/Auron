# =============================================================
# main.py — Entry point (Phase 3: persistent memory commands)
#
# NEW COMMANDS IN PHASE 3:
#   remember: <text>   — Save something to persistent memory
#   memories           — Show everything the agent remembers
#   forget <id>        — Delete a specific memory by its short ID
#   clear-memories     — Delete ALL memories (asks for confirmation)
# =============================================================

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from agent.core import Agent
from agent.config import get_config

console = Console()


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]AI Agent[/bold cyan] [dim]v2.0 — Phase 3[/dim]\n"
        "[dim]Type your message and press Enter[/dim]\n"
        "[dim]Commands: exit | clear | stats | help | memories | forget <id>[/dim]\n"
        "[dim]Memory:   remember: <text to save>[/dim]",
        border_style="cyan",
    ))


def print_help() -> None:
    console.print(Panel(
        "[bold]Chat commands:[/bold]\n"
        "  [cyan]exit[/cyan]              — Quit the agent\n"
        "  [cyan]clear[/cyan]             — Clear conversation history (keeps memories)\n"
        "  [cyan]stats[/cyan]             — Show token usage and session info\n"
        "  [cyan]help[/cyan]              — Show this message\n\n"
        "[bold]Memory commands:[/bold]\n"
        "  [cyan]remember: <text>[/cyan]  — Save something to persistent memory\n"
        "                         Example: remember: I always use Black formatter\n"
        "  [cyan]memories[/cyan]          — Show everything the agent remembers\n"
        "  [cyan]forget <id>[/cyan]       — Delete a memory by its short ID\n"
        "                         Example: forget a1b2c3d4\n"
        "  [cyan]clear-memories[/cyan]    — Delete ALL memories (asks for confirmation)\n\n"
        "[dim]Everything else is sent to the agent as a message.[/dim]",
        title="Help",
        border_style="dim",
    ))


def print_stats(agent: Agent) -> None:
    stats = agent.get_stats()
    config = agent.config
    memory_count = len(agent.memory.get_all())
    console.print(Panel(
        f"[bold]Session Statistics[/bold]\n\n"
        f"  LLM calls:        [cyan]{stats.total_llm_calls}[/cyan]\n"
        f"  Tool calls:       [cyan]{stats.total_tool_calls}[/cyan]\n"
        f"  Input tokens:     [cyan]{stats.total_input_tokens:,}[/cyan]\n"
        f"  Output tokens:    [cyan]{stats.total_output_tokens:,}[/cyan]\n"
        f"  Total tokens:     [cyan]{stats.total_tokens:,}[/cyan]\n"
        f"  Session time:     [cyan]{stats.session_duration_seconds:.0f}s[/cyan]\n"
        f"  Memories saved:   [cyan]{memory_count}[/cyan]\n\n"
        f"[bold]Configuration[/bold]\n\n"
        f"  Provider:   [cyan]{config.model.provider}[/cyan]\n"
        f"  Model:      [cyan]{config.model.name}[/cyan]\n"
        f"  Project:    [cyan]{config.memory.project}[/cyan]\n"
        f"  Memory:     [cyan]{config.memory.backend}[/cyan]\n"
        f"  Budget:     [cyan]{config.rate_limits.max_tokens_per_session:,} tokens/session[/cyan]",
        title="Stats",
        border_style="dim",
    ))


async def handle_remember(agent: Agent, raw_input: str) -> None:
    content_raw = raw_input[len("remember:"):].strip()
    if not content_raw:
        console.print("[yellow]Nothing to remember. Usage: remember: <text>[/yellow]")
        return
    content, category = agent.memory.extract_memory_content(content_raw)
    if not content:
        console.print("[yellow]Could not extract content to remember.[/yellow]")
        return
    try:
        memory = agent.memory.save(content, category=category)
        console.print(
            f"[green]✓ Remembered[/green] [{category}]: {content}\n"
            f"[dim]  ID: {memory.memory_id[:8]} — use 'forget {memory.memory_id[:8]}' to delete[/dim]"
        )
    except Exception as e:
        console.print(f"[red]Failed to save memory: {e}[/red]")


async def handle_memories(agent: Agent) -> None:
    formatted = agent.memory.format_memories_for_display()
    console.print(Panel(formatted, title="Persistent Memories", border_style="cyan"))


async def handle_forget(agent: Agent, raw_input: str) -> None:
    parts = raw_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        console.print("[yellow]Usage: forget <id>  (the 8-character ID shown in 'memories')[/yellow]")
        return
    short_id = parts[1].strip()
    all_memories = agent.memory.get_all()
    matches = [m for m in all_memories if m.memory_id.startswith(short_id)]
    if not matches:
        console.print(f"[yellow]No memory found with ID starting with '{short_id}'[/yellow]")
        return
    if len(matches) > 1:
        console.print(f"[yellow]Multiple memories match '{short_id}'. Use more characters.[/yellow]")
        return
    memory = matches[0]
    deleted = agent.memory.delete(memory.memory_id)
    if deleted:
        console.print(f"[green]✓ Forgotten:[/green] {memory.content[:80]}")
    else:
        console.print(f"[red]Failed to delete memory {short_id}[/red]")


async def handle_clear_memories(agent: Agent) -> None:
    count = len(agent.memory.get_all())
    if count == 0:
        console.print("[dim]No memories to clear.[/dim]")
        return
    console.print(f"[yellow]⚠️  This will permanently delete all {count} memories.[/yellow]")
    answer = console.input("Are you sure? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        deleted = agent.memory.clear_all()
        console.print(f"[green]✓ Cleared {deleted} memories.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


async def chat_loop() -> None:
    try:
        config = get_config()
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)

    print_banner()
    console.print(
        f"[dim]Provider: {config.model.provider} | "
        f"Model: {config.model.name} | "
        f"Memory: {config.memory.backend} | "
        f"Project: {config.memory.project}[/dim]\n"
    )

    try:
        agent = Agent(config=config, project=config.memory.project)
    except Exception as e:
        console.print(f"[bold red]Failed to create agent:[/bold red] {e}")
        sys.exit(1)

    memory_count = len(agent.memory.get_all())
    if memory_count > 0:
        console.print(
            f"[dim]💾 {memory_count} memories loaded from previous sessions. "
            f"Type 'memories' to see them.[/dim]\n"
        )

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]You:[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input_stripped = user_input.strip()
        if not user_input_stripped:
            continue

        lower = user_input_stripped.lower()

        if lower == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        elif lower == "clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared. Memories are kept.[/dim]")
            continue
        elif lower == "stats":
            print_stats(agent)
            continue
        elif lower == "help":
            print_help()
            continue
        elif lower == "memories":
            await handle_memories(agent)
            continue
        elif lower.startswith("remember:"):
            await handle_remember(agent, user_input_stripped)
            continue
        elif lower.startswith("forget "):
            await handle_forget(agent, user_input_stripped)
            continue
        elif lower == "clear-memories":
            await handle_clear_memories(agent)
            continue

        # Regular chat message
        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = await agent.chat(user_input_stripped)

            console.print()
            console.print(Text("Agent:", style="bold blue"), end=" ")
            console.print(Markdown(response.content))
            console.print(
                f"[dim]  ↳ {response.input_tokens} in / "
                f"{response.output_tokens} out / "
                f"{response.duration_ms:.0f}ms[/dim]"
            )

            # Auto-detect memorable content and offer to save it
            if agent.memory.should_auto_save(user_input_stripped):
                content, _ = agent.memory.extract_memory_content(user_input_stripped)
                console.print(
                    f"[dim]  💡 Tip: Type 'remember: {content[:50]}' to save this for future sessions.[/dim]"
                )

        except RuntimeError as e:
            console.print(f"[bold yellow]⚠ {e}[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("[dim]The conversation history is intact. Try again.[/dim]")


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
