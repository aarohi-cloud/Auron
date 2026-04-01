# =============================================================
# agent/config.py — Reads, validates, and exposes configuration
#
# WHAT THIS FILE DOES:
#   1. Defines the exact shape/structure of every config value
#   2. Reads agent.yaml on startup
#   3. Allows environment variables to override any value
#   4. Validates everything — crashes early with a clear message
#      if something is wrong, rather than crashing mysteriously later
#
# HOW TO USE IT (from any other file):
#   from agent.config import get_config
#   config = get_config()
#   print(config.model.name)  # prints "claude-haiku-4-5-20251001"
# =============================================================

# 'os' lets us read environment variables (like ANTHROPIC_API_KEY)
import os

# 'Path' is a clean way to work with file paths
from pathlib import Path

# 'Optional' means a value might be None (not set)
# 'Literal' means a value must be exactly one of a fixed set of options
from typing import Optional, Literal

# 'yaml' reads .yaml files and converts them to Python dictionaries
import yaml

# 'pydantic' is our validation library
# BaseModel = a class whose fields are automatically validated
# Field = lets us add rules to individual fields (min value, max value, etc.)
# field_validator = a decorator to write custom validation logic
from pydantic import BaseModel, Field, field_validator

# 'load_dotenv' reads the .env file and puts its contents into
# environment variables so Python can access them via os.environ
from dotenv import load_dotenv

# Load the .env file as soon as this module is imported.
# This means ANTHROPIC_API_KEY etc. are available immediately.
load_dotenv()


# =============================================================
# SECTION MODELS
# Each class below represents one section of agent.yaml
# Think of each class as a form with typed fields
# =============================================================

class ModelConfig(BaseModel):
    """Settings for which AI model to use and how."""

    # Literal means this field MUST be one of these exact strings.
    # If someone writes provider: "gpt-banana" in agent.yaml, pydantic
    # will immediately raise an error explaining the valid options.
    provider: Literal["anthropic", "ollama", "openai_compatible"] = "anthropic"

    name: str = "claude-haiku-4-5-20251001"

    # Field(...) lets us add constraints.
    # ge=1 means "greater than or equal to 1"
    # le=200000 means "less than or equal to 200000"
    max_tokens: int = Field(default=4096, ge=1, le=200000)

    # ge=0.0, le=2.0 — temperature must be between 0 and 2
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    ollama_base_url: str = "http://localhost:11434"
    openai_compatible_base_url: str = ""

    # This is a "property" — it looks like a field but is computed
    # on the fly. We use it to get the API key from environment variables
    # (never from the YAML file, because YAML might be committed to Git)
    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment variable — never from config file."""
        if self.provider == "anthropic":
            # os.environ.get returns None if the variable doesn't exist
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "openai_compatible":
            return os.environ.get("OPENAI_API_KEY")
        # Ollama runs locally and doesn't need an API key
        return None


class MemoryConfig(BaseModel):
    """Settings for how the agent stores memories."""

    backend: Literal["in_memory", "chroma"] = "chroma"
    path: str = "./data/memory"
    top_k_results: int = Field(default=5, ge=1, le=50)
    # Project namespace — keeps memories from different projects separate
    project: str = "default"
    # Minimum similarity score (0.0–1.0) for a memory to be retrieved
    # ChromaDB returns a distance score; we convert it to similarity
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class ShellToolConfig(BaseModel):
    """Settings for the shell (terminal command) tool."""
    enabled: bool = True
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class FileIOToolConfig(BaseModel):
    """Settings for the file read/write tool."""
    enabled: bool = True
    allowed_base_path: str = "."


class WebSearchToolConfig(BaseModel):
    """Settings for the web search tool."""
    enabled: bool = True
    timeout_seconds: int = Field(default=15, ge=1, le=60)


class WebBrowseToolConfig(BaseModel):
    """Settings for the web browsing tool."""
    enabled: bool = True
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class ToolsConfig(BaseModel):
    """Container for all tool configurations."""
    shell: ShellToolConfig = ShellToolConfig()
    file_io: FileIOToolConfig = FileIOToolConfig()
    web_search: WebSearchToolConfig = WebSearchToolConfig()
    web_browse: WebBrowseToolConfig = WebBrowseToolConfig()


class SafetyConfig(BaseModel):
    """Settings for safety and human-in-the-loop behaviour."""
    require_confirmation_for_destructive_actions: bool = True
    auto_approve_actions: list[str] = ["read_file", "web_search", "list_directory"]
    prompt_injection_detection: bool = True


class RateLimitsConfig(BaseModel):
    """Settings to prevent runaway API usage and costs."""
    max_llm_calls_per_minute: int = Field(default=20, ge=1)
    max_tool_calls_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_task: int = Field(default=50000, ge=0)
    max_tokens_per_session: int = Field(default=200000, ge=0)


class ObservabilityConfig(BaseModel):
    """Settings for logging and monitoring."""

    # Literal ensures only valid log levels are accepted
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "./logs/agent.log"
    structured_logging: bool = True
    otel_endpoint: str = ""


class AgentBehaviourConfig(BaseModel):
    """Settings for how the agent thinks and behaves."""
    name: str = "AI Agent"
    max_self_correction_attempts: int = Field(default=3, ge=1, le=10)
    max_planning_depth: int = Field(default=10, ge=1, le=50)
    show_thinking: bool = True


# =============================================================
# ROOT CONFIG MODEL
# This is the top-level object that holds ALL sections.
# When you call get_config(), you get back one of these.
# =============================================================

class AgentConfig(BaseModel):
    """
    The complete, validated configuration for the AI Agent.

    This is the single source of truth for all settings.
    All sections of agent.yaml map to fields in this class.
    """

    # 'version' helps us support config file upgrades in future
    version: int = 2

    # Each field here corresponds to a top-level section in agent.yaml
    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()
    safety: SafetyConfig = SafetyConfig()
    rate_limits: RateLimitsConfig = RateLimitsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    agent: AgentBehaviourConfig = AgentBehaviourConfig()

    # field_validator runs custom validation logic after pydantic
    # has already validated individual field types.
    # mode="after" means it runs after all individual fields are validated.
    @field_validator("model", mode="after")
    @classmethod
    def validate_api_key_present(cls, model_config: ModelConfig) -> ModelConfig:
        """
        Check that an API key exists when using a cloud provider.
        We don't check Ollama because it runs locally without a key.
        """
        if model_config.provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                # Raise a clear, actionable error message
                raise ValueError(
                    "\n\n❌ ANTHROPIC_API_KEY is not set!\n"
                    "   Fix: Add this line to your .env file:\n"
                    "   ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
                )
        return model_config


# =============================================================
# CONFIG LOADER
# This function loads the YAML file and builds an AgentConfig
# =============================================================

# Module-level cache: we only load the config file ONCE.
# After the first call, we return the cached version instantly.
# The underscore prefix is a Python convention meaning "private variable"
_config_cache: Optional[AgentConfig] = None


def load_config(config_path: str = "agent.yaml") -> AgentConfig:
    """
    Load and validate configuration from agent.yaml.

    This function:
    1. Reads the YAML file
    2. Converts it to a Python dictionary
    3. Pydantic validates every field
    4. Returns a fully validated AgentConfig object

    Args:
        config_path: Path to the YAML config file.
                     Default is "agent.yaml" in the current directory.

    Returns:
        AgentConfig: The complete, validated config object.

    Raises:
        FileNotFoundError: If agent.yaml doesn't exist
        ValidationError: If any config value is invalid (pydantic)
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n\n❌ Config file not found: {config_path}\n"
            f"   Make sure agent.yaml exists in your project root.\n"
        )

    # Read the YAML file
    # 'with open(...)' safely opens a file and closes it automatically
    # 'r' means "read mode" (not write)
    # 'encoding="utf-8"' handles special characters properly
    with open(path, "r", encoding="utf-8") as f:
        # yaml.safe_load converts YAML text into a Python dictionary
        # 'safe_load' (not 'load') prevents code execution in YAML files
        raw_data = yaml.safe_load(f)

    # If the file is empty, use an empty dict
    # The 'or {}' handles the case where safe_load returns None
    raw_data = raw_data or {}

    # Pass the dictionary to AgentConfig.
    # Pydantic will validate every field and raise a clear error
    # if anything is wrong — before the agent does anything.
    # The '**' operator "unpacks" the dict as keyword arguments.
    config = AgentConfig(**raw_data)

    return config


def get_config(config_path: str = "agent.yaml") -> AgentConfig:
    """
    Get the global config, loading it only once (cached).

    This is the function you call from ALL other files.
    The first call loads the file. Every call after returns
    the already-loaded version instantly (no re-reading the file).

    Usage:
        from agent.config import get_config
        config = get_config()
        print(config.model.name)
    """
    # 'global' lets us modify the module-level variable
    global _config_cache

    # If we haven't loaded config yet, load it now
    if _config_cache is None:
        _config_cache = load_config(config_path)

    return _config_cache


def reset_config() -> None:
    """
    Clear the config cache. Used in tests so each test
    gets a fresh config rather than the cached one.
    """
    global _config_cache
    _config_cache = None
