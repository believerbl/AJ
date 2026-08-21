import asyncio
import logging
import sys

from core.orchestrator import AgenticOrchestrator
from learning.idle_trainer import IdleTrainer

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("JarvisMain")

async def main():
    logger.info("Starting Jarvis Autonomous Local AI Agent...")
    
    # Initialize Core System
    orchestrator = AgenticOrchestrator()
    
    # Initialize and start Idle Monitor / Trainer
    idle_monitor = IdleTrainer(orchestrator=orchestrator)
    idle_monitor.run_monitor()
    
    print("\n" + "="*50)
    print("Jarvis is online and ready.")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                logger.info("Shutting down Jarvis...")
                break
                
            if not user_input.strip():
                continue
                
            response = await orchestrator.process_user_input(user_input)
            print(f"\nJarvis: {response}\n")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
