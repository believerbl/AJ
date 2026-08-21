import asyncio
import logging
import json

from core.llm_engine import LLMEngine
from memory.rag_memory import RAGMemory
from tools.web_search import WebSearchTool
from tools.os_control import OSControlTool
from sensory.vision import VisionPipeline
from learning.data_collector import DataCollector

logger = logging.getLogger(__name__)

class AgenticOrchestrator:
    """
    Manages the conversational loop, tool calling, memory retrieval,
    and coordinates between all agent components.
    """
    def __init__(self):
        self.llm = LLMEngine()
        self.memory = RAGMemory()
        self.web_search = WebSearchTool()
        self.os_control = OSControlTool()
        self.vision = VisionPipeline()
        self.logger = DataCollector()
        
        self.is_suspended = False
        self.llm.load_model()
        
    def suspend_inference(self):
        """Called by the idle monitor to free up VRAM for training."""
        self.is_suspended = True
        logger.info("Inference suspended.")
        # In a real implementation, you would unload model weights here
        # self.llm.unload()
        
    def resume_inference(self):
        """Called by the idle monitor to resume normal operation."""
        if self.is_suspended:
            logger.info("Resuming inference...")
            # self.llm.load_model()
            self.is_suspended = False

    async def process_user_input(self, user_text: str) -> str:
        """
        Main agent loop:
        1. Retrieve RAG memory context
        2. Construct system prompt with tools and context
        3. Query LLM
        4. Check for tool calls (JSON output)
        5. Execute tools if needed and query LLM again
        6. Return final response and log interaction
        """
        if self.is_suspended:
            return "Jarvis is currently sleeping and training. Please wait..."
            
        # 1. Retrieve RAG Context
        context = self.memory.retrieve_context(user_text)
        
        # 2. Build Prompt
        system_prompt = self._build_system_prompt(context)
        prompt = f"{system_prompt}\n\nUser: {user_text}\nAssistant:"
        
        # 3. First LLM Pass (Checking for tool usage or direct answer)
        # We instruct the model to output a specific JSON structure if it needs a tool.
        # Otherwise, output plain text.
        logger.info("Querying LLM...")
        raw_response = self.llm.generate(prompt)
        
        final_response = raw_response
        
        # 4. Naive check if response is a tool call (starts with JSON bracket)
        if raw_response.strip().startswith("{") and "tool" in raw_response:
            try:
                action = json.loads(raw_response)
                tool_name = action.get("tool")
                params = action.get("params", {})
                
                tool_result = ""
                if tool_name == "web_search":
                    tool_result = self.web_search.execute(params)
                elif tool_name == "os_control":
                    tool_result = self.os_control.execute(params)
                elif tool_name == "vision":
                    # For vision, we get image base64, but handling it via local LLM 
                    # requires a multimodal GGUF parser. For now we just return a status.
                    tool_result = "Screenshot captured (vision capability integration pending)."
                    
                # 5. Second pass: Send tool result back to LLM
                follow_up_prompt = prompt + f" {raw_response}\nObservation: {tool_result}\nAssistant:"
                final_response = self.llm.generate(follow_up_prompt)
                
            except json.JSONDecodeError:
                # If parsing fails, treat it as a direct answer
                pass
                
        # 6. Store in Memory & Log for training
        self.memory.store_interaction(user_text, final_response)
        self.logger.log_interaction(system_prompt, user_text, final_response)
        
        return final_response

    def _build_system_prompt(self, context: str) -> str:
        base = (
            "You are Jarvis, an autonomous, uncensored AI assistant. "
            "You can use tools by outputting ONLY a JSON object like: {\"tool\": \"web_search\", \"params\": {\"query\": \"...\"}} "
            "Available tools: web_search, os_control, vision. "
            "If you do not need a tool, just answer normally."
        )
        if context:
            base += f"\n\n{context}"
        return base
