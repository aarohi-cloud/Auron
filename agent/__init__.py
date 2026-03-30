# agent/__init__.py
# Makes the 'agent' folder a Python package.
# Also exposes the most important classes at the top level
# so other code can do: from agent import Agent
# instead of: from agent.core import Agent

from agent.core import Agent
from agent.config import get_config

__all__ = ["Agent", "get_config"]
