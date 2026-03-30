# =============================================================
# agent/tools/web_search.py — Web search and page reading
#
# TOOLS PROVIDED:
#   web_search  — Search the web (DuckDuckGo, no API key needed)
#   read_url    — Fetch and read the content of a specific URL
#
# WHY DUCKDUCKGO:
#   - Completely free — no API key, no account, no limits
#   - Privacy-respecting
#   - Returns clean results
#   - The 'duckduckgo-search' library makes it simple
#
# WHAT THIS ENABLES:
#   "What does this error mean?" → searches Stack Overflow
#   "Check the FastAPI docs for X" → reads fastapi.tiangolo.com
#   "Are there any CVEs in requests 2.28?" → searches security advisories
#   "What's new in Python 3.13?" → reads the Python changelog
# =============================================================

import asyncio
from typing import Any
from urllib.parse import urlparse

import structlog

from agent.tools.base import BaseTool, ToolResult
from agent.config import get_config

logger = structlog.get_logger(__name__)

# Maximum characters to return from a web page
# Web pages can be enormous — we cap it to avoid flooding the AI's context
MAX_PAGE_CHARS = 8000


class WebSearchTool(BaseTool):
    """
    Search the web using DuckDuckGo and return results.

    Returns a list of search results with titles, URLs, and snippets.
    The AI can then use ReadURLTool to fetch any specific result.
    """

    def __init__(self):
        config = get_config()
        self._timeout = config.tools.web_search.timeout_seconds

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo and return the top results. "
            "Use this to find documentation, look up error messages, check library "
            "changelogs, search for CVEs, find Stack Overflow answers, or get "
            "any other information from the internet. "
            "Returns titles, URLs, and snippets. Use read_url to fetch full page content. "
            "No API key required."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific for better results. "
                        "Examples: 'FastAPI dependency injection tutorial', "
                        "'Python asyncio TimeoutError fix', "
                        "'requests library CVE 2024'"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 5.",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["query"])
        if error:
            return ToolResult.fail(error)

        query = arguments["query"].strip()
        max_results = int(arguments.get("max_results", 5))

        if not query:
            return ToolResult.fail("Search query cannot be empty.")

        logger.info("web_search_starting", query=query, max_results=max_results)

        try:
            # Import here so missing library gives a clear error message
            # instead of failing on module import
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return ToolResult.fail(
                    "The 'duckduckgo-search' package is not installed.\n"
                    "Fix: run 'pip install duckduckgo-search' then restart."
                )

            # DuckDuckGo search is synchronous (blocking), so we run it
            # in a thread pool so it doesn't block our async event loop.
            # asyncio.get_event_loop().run_in_executor() runs a blocking
            # function in a background thread, freeing up the main thread.
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,  # None = use default thread pool
                lambda: list(DDGS().text(query, max_results=max_results))
            )

            if not results:
                return ToolResult.ok(
                    f"No results found for: '{query}'\n"
                    f"Try rephrasing your search query."
                )

            # Format results in a way the AI can easily parse
            formatted_parts = [f"Search results for: '{query}'\n"]

            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("href", "")
                snippet = result.get("body", "No description available")

                # Truncate very long snippets
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."

                formatted_parts.append(
                    f"{i}. {title}\n"
                    f"   URL: {url}\n"
                    f"   {snippet}\n"
                )

            output = "\n".join(formatted_parts)

            logger.info("web_search_complete", query=query, result_count=len(results))

            return ToolResult.ok(
                output=output,
                metadata={"query": query, "result_count": len(results)}
            )

        except Exception as e:
            logger.error("web_search_error", query=query, error=str(e))
            return ToolResult.fail(
                f"Web search failed: {e}\n"
                f"Check your internet connection and try again."
            )


class ReadURLTool(BaseTool):
    """
    Fetch the content of a specific URL and return it as text.

    The AI uses this after web_search to read full documentation pages,
    Stack Overflow answers, GitHub issues, changelogs, etc.
    """

    def __init__(self):
        config = get_config()
        self._timeout = config.tools.web_browse.timeout_seconds

    @property
    def name(self) -> str:
        return "read_url"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and return its text content. "
            "Use this to read full documentation pages, Stack Overflow answers, "
            "GitHub READMEs, changelogs, blog posts, or any public web page. "
            "HTML tags are stripped — you get clean readable text. "
            f"Returns up to {MAX_PAGE_CHARS} characters. "
            "Does not work with pages that require login."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The full URL to fetch. Must start with http:// or https://. "
                        "Example: 'https://docs.python.org/3/library/asyncio.html'"
                    )
                }
            },
            "required": ["url"]
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        error = self._validate_required_args(arguments, ["url"])
        if error:
            return ToolResult.fail(error)

        url = arguments["url"].strip()

        # Basic URL validation
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ToolResult.fail(
                    f"Invalid URL: '{url}'. Must start with http:// or https://"
                )
        except Exception:
            return ToolResult.fail(f"Could not parse URL: '{url}'")

        logger.info("read_url_starting", url=url)

        try:
            try:
                import httpx            # async HTTP client
                from bs4 import BeautifulSoup  # HTML parser
            except ImportError as e:
                missing = str(e).split("'")[1]
                return ToolResult.fail(
                    f"Required package '{missing}' is not installed.\n"
                    f"Fix: pip install httpx beautifulsoup4"
                )

            # Use httpx for async HTTP requests
            # 'follow_redirects=True' handles 301/302 redirects automatically
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self._timeout,
                headers={
                    # Identify ourselves — some sites block requests without a user agent
                    "User-Agent": "AI-Agent/2.0 (educational project)"
                }
            ) as client:
                response = await client.get(url)
                response.raise_for_status()  # Raises error for 4xx/5xx status codes

            # Parse HTML and extract plain text
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements — we don't want JavaScript code
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text, collapse whitespace
            text = soup.get_text(separator="\n")

            # Clean up: remove excessive blank lines
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]  # remove empty lines
            clean_text = "\n".join(lines)

            # Truncate if too long
            if len(clean_text) > MAX_PAGE_CHARS:
                clean_text = clean_text[:MAX_PAGE_CHARS]
                clean_text += f"\n\n[Content truncated at {MAX_PAGE_CHARS} characters]"

            logger.info(
                "read_url_complete",
                url=url,
                content_length=len(clean_text),
                status_code=response.status_code,
            )

            return ToolResult.ok(
                output=f"Content from: {url}\n\n{clean_text}",
                metadata={"url": url, "status_code": response.status_code}
            )

        except Exception as e:
            # httpx raises specific exceptions we can give good messages for
            error_msg = str(e)

            if "ConnectError" in type(e).__name__ or "ConnectTimeout" in type(e).__name__:
                return ToolResult.fail(
                    f"Could not connect to: {url}\n"
                    f"Check the URL and your internet connection."
                )
            elif "HTTPStatusError" in type(e).__name__:
                return ToolResult.fail(
                    f"HTTP error fetching {url}: {error_msg}"
                )
            else:
                return ToolResult.fail(f"Error fetching URL: {error_msg}")
