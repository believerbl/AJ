import time
import logging
import threading

import config

try:
    from pynput import mouse, keyboard
except ImportError:
    mouse = None
    keyboard = None

logger = logging.getLogger(__name__)

class IdleTrainer:
    """
    Monitors OS idle time and triggers background LoRA training
    on the collected dataset when the machine is inactive.
    """
    def __init__(self, orchestrator=None):
        self.last_activity_time = time.time()
        self.is_idle = False
        self.training_thread = None
        self.stop_training_flag = threading.Event()
        self.orchestrator = orchestrator # To pause LLM inference
        
        if mouse and keyboard:
            self.mouse_listener = mouse.Listener(on_move=self.on_activity, on_click=self.on_activity, on_scroll=self.on_activity)
            self.keyboard_listener = keyboard.Listener(on_press=self.on_activity)
            self.mouse_listener.start()
            self.keyboard_listener.start()
            logger.info("Idle monitor started.")
        else:
            logger.warning("pynput not installed. Idle monitor disabled.")

    def on_activity(self, *args):
        """Callback for any input activity."""
        self.last_activity_time = time.time()
        if self.is_idle:
            self.is_idle = False
            logger.info("User activity detected. Waking up...")
            self.interrupt_training()
            if self.orchestrator:
                self.orchestrator.resume_inference()

    def check_idle_status(self):
        """Loop that runs continuously to check if timeout is reached."""
        while True:
            time.sleep(5)
            if not self.is_idle and (time.time() - self.last_activity_time) > config.IDLE_TIMEOUT_SECONDS:
                self.is_idle = True
                logger.info(f"Machine idle for {config.IDLE_TIMEOUT_SECONDS}s. Suspending inference and starting training.")
                if self.orchestrator:
                    self.orchestrator.suspend_inference()
                self.start_training()

    def start_training(self):
        """Starts the PyTorch LoRA training loop in a background thread."""
        if not config.DATASET_PATH.exists():
            logger.info("No dataset found for training. Skipping.")
            return

        self.stop_training_flag.clear()
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()

    def _training_loop(self):
        """
        The actual PyTorch/PEFT training loop.
        Note: This is a scaffold. Full PyTorch integration requires 
        loading model weights with transformers and running trainer.train().
        """
        logger.info("--- STARTED BACKGROUND LoRA TRAINING ---")
        
        # Simulated training loop for demonstration
        # In reality, you would setup transformers.Trainer here and use a callback
        # to check self.stop_training_flag.is_set() at each step.
        steps = 0
        while not self.stop_training_flag.is_set() and steps < 100:
            time.sleep(1) # Simulate a training step
            steps += 1
            if steps % 10 == 0:
                logger.info(f"Training step {steps}...")
                
        if self.stop_training_flag.is_set():
            logger.info("--- TRAINING INTERRUPTED BY USER ACTIVITY ---")
        else:
            logger.info("--- TRAINING COMPLETED ---")
            
    def interrupt_training(self):
        """Interrupts training gracefully if running."""
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training_flag.set()
            self.training_thread.join(timeout=5.0)
            self.training_thread = None

    def run_monitor(self):
        """Starts the check loop in a daemon thread."""
        monitor_thread = threading.Thread(target=self.check_idle_status, daemon=True)
        monitor_thread.start()
