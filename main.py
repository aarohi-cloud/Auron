# =============================================================
# main.py — Entry point. Run this to start the agent.
#
# USAGE:
#   python main.py
#
# This starts an interactive chat loop in your terminal.
# Type your message and press Enter. Type 'exit' to quit.
# =============================================================

import asyncio
import sys
from pathlib import Path

# 'rich' gives us beautiful, coloured terminal output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

# Our agent modules
from agent.core import Agent
from agent.config import get_config

# Create a rich Console — this is how we print to the terminal
# with colours, formatting, etc.
console = Console()


def print_banner() -> None:
    """Print the startup banner."""
    console.print(Panel.fit(
        "[bold cyan]AI Agent[/bold cyan] [dim]v2.0[/dim]\n"
        "[dim]Type your message and press Enter[/dim]\n"
        "[dim]Commands: 'exit' | 'clear' | 'stats' | 'help'[/dim]",
        border_style="cyan",
    ))


def print_help() -> None:
    """Print available commands."""
    console.print(Panel(
        "[bold]Available commands:[/bold]\n\n"
        "  [cyan]exit[/cyan]   — Quit the agent\n"
        "  [cyan]clear[/cyan]  — Clear conversation history and start fresh\n"
        "  [cyan]stats[/cyan]  — Show token usage and session statistics\n"
        "  [cyan]help[/cyan]   — Show this help message\n\n"
        "[dim]Everything else is sent to the AI as a message.[/dim]",
        title="Help",
        border_style="dim",
    ))


def print_stats(agent: Agent) -> None:
    """Print current session statistics."""
    stats = agent.get_stats()
    config = agent.config

    console.print(Panel(
        f"[bold]Session Statistics[/bold]\n\n"
        f"  LLM calls:      [cyan]{stats.total_llm_calls}[/cyan]\n"
        f"  Input tokens:   [cyan]{stats.total_input_tokens:,}[/cyan]\n"
        f"  Output tokens:  [cyan]{stats.total_output_tokens:,}[/cyan]\n"
        f"  Total tokens:   [cyan]{stats.total_tokens:,}[/cyan]\n"
        f"  Session time:   [cyan]{stats.session_duration_seconds:.0f}s[/cyan]\n\n"
        f"[bold]Configuration[/bold]\n\n"
        f"  Provider:  [cyan]{config.model.provider}[/cyan]\n"
        f"  Model:     [cyan]{config.model.name}[/cyan]\n"
        f"  Budget:    [cyan]{config.rate_limits.max_tokens_per_session:,} tokens/session[/cyan]",
        title="Stats",
        border_style="dim",
    ))


async def chat_loop() -> None:
    """
    The main interactive chat loop.

    This runs forever, reading input from the terminal and
    sending it to the agent, until the user types 'exit'.
    """
    # Load config — this reads agent.yaml and validates it
    # If something is wrong with your config, you'll see a clear error here
    try:
        config = get_config()
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)

    # Print the startup banner
    print_banner()

    console.print(
        f"[dim]Provider: {config.model.provider} | "
        f"Model: {config.model.name}[/dim]\n"
    )

    # Create the agent
    try:
        agent = Agent(config=config)
    except Exception as e:
        console.print(f"[bold red]Failed to create agent:[/bold red] {e}")
        sys.exit(1)

    # The main loop — keeps running until user types 'exit'
    while True:
        try:
            # Print the input prompt and wait for user to type something
            # [bold green]You:[/bold green] makes "You:" appear in bold green
            console.print()  # blank line for readability
            user_input = console.input("[bold green]You:[/bold green] ")

        except (KeyboardInterrupt, EOFError):
            # User pressed Ctrl+C or Ctrl+D — exit gracefully
            console.print("\n[dim]Goodbye![/dim]")
            break

        # Strip whitespace from both ends of the input
        user_input = user_input.strip()

        # Handle empty input — just show the prompt again
        if not user_input:
            continue

        # Handle special commands
        if user_input.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input.lower() == "clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared. Starting fresh.[/dim]")
            continue

        elif user_input.lower() == "stats":
            print_stats(agent)
            continue

        elif user_input.lower() == "help":
            print_help()
            continue

        # Send the message to the agent
        try:
            # Show a "thinking" indicator while waiting for the AI
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = await agent.chat(user_input)

            # Print the agent's response
            # We print it as Markdown so code blocks, bold, etc. are formatted
            console.print()
            console.print(Text("Agent:", style="bold blue"), end=" ")
            console.print(Markdown(response.content))

            # Show token usage for this response
            console.print(
                f"[dim]  ↳ {response.input_tokens} in / "
                f"{response.output_tokens} out / "
                f"{response.duration_ms:.0f}ms[/dim]"
            )

        except RuntimeError as e:
            # Rate limit or token budget exceeded
            console.print(f"[bold yellow]⚠ {e}[/bold yellow]")

        except Exception as e:
            # Unexpected error — show it clearly
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("[dim]The conversation history is intact. Try again.[/dim]")


def main() -> None:
    """
    The entry point function.
    'asyncio.run()' starts the async event loop and runs chat_loop().
    """
    asyncio.run(chat_loop())


# This is a Python convention:
# This block only runs when you execute this file directly
# (python main.py), NOT when it's imported by another file.
if __name__ == "__main__":
    main()