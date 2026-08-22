import logging
from typing import List, Dict

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)


def run_web_search(query: str, max_results: int = 4) -> str:
    """
    Search DuckDuckGo and return a formatted string of the top results.
    No API key required. Designed to be called directly by tool_node with
    the flat {"tool": "web_search", "input": "<query>"} schema.
    """
    if DDGS is None:
        return "Error: duckduckgo-search is not installed. Run: pip install duckduckgo-search"

    if not query.strip():
        return "Error: empty search query."

    logger.info(f"[web_search] querying: {query}")

    try:
        results: List[Dict] = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        logger.error(f"[web_search] failed: {e}")
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    {r.get('body', 'No snippet')}\n"
            f"    {r.get('href', '')}"
        )

    return "\n\n".join(lines)
