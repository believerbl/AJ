import logging
from typing import List, Dict, Any

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Tool for performing real-time web searches using DuckDuckGo.
    Does not require an API key and operates locally to fetch results.
    """
    def __init__(self):
        self.ddgs = DDGS() if DDGS else None

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Perform a web search and return a list of results.
        Each result contains 'title', 'link', and 'snippet' (body).
        """
        if self.ddgs is None:
            logger.error("duckduckgo-search is not installed.")
            return [{"error": "Search tool unavailable."}]

        logger.info(f"Searching web for: {query}")
        results = []
        try:
            # text() returns an iterator of dictionaries
            search_results = self.ddgs.text(query, max_results=max_results)
            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"error": str(e)}]
        
    def execute(self, params: Dict[str, Any]) -> str:
        """
        Execute wrapper for agentic JSON calling.
        Expects params {"query": "..."}
        """
        query = params.get("query", "")
        if not query:
            return "Error: No query provided."
            
        results = self.search(query)
        
        # Format results into a string for the LLM
        if not results or "error" in results[0]:
            return f"Search failed or no results. {results[0].get('error', '')}"
            
        formatted_results = []
        for i, res in enumerate(results, 1):
            formatted_results.append(f"Result {i}:\nTitle: {res['title']}\nSnippet: {res['snippet']}\nLink: {res['link']}")
            
        return "\n\n".join(formatted_results)
