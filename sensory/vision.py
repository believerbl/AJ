import logging
import pyautogui
from PIL import Image
import io
import base64

import config

logger = logging.getLogger(__name__)

class VisionPipeline:
    """
    Handles capturing desktop screenshots and formatting them for the LLM.
    """
    def __init__(self):
        # Prevent PyAutoGUI from causing a fail-safe exception accidentally
        pyautogui.FAILSAFE = False

    def capture_screen(self) -> Image.Image:
        """
        Capture the main screen.
        """
        try:
            screenshot = pyautogui.screenshot()
            return screenshot
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

    def process_image_for_llm(self, image: Image.Image) -> str:
        """
        Resize, compress, and encode the image in base64.
        """
        if not image:
            return ""

        try:
            # Resize image based on config to save tokens/VRAM
            new_size = (
                int(image.width * config.SCREENSHOT_RESIZE_FACTOR),
                int(image.height * config.SCREENSHOT_RESIZE_FACTOR)
            )
            
            # Ensure it doesn't exceed maximum bounds
            if new_size[0] > config.SCREENSHOT_MAX_SIZE[0] or new_size[1] > config.SCREENSHOT_MAX_SIZE[1]:
                image.thumbnail(config.SCREENSHOT_MAX_SIZE, Image.Resampling.LANCZOS)
            else:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
            # Convert to RGB (in case of RGBA)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Compress to JPEG
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=70)
            
            # Encode base64
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return ""

    def get_vision_context(self) -> str:
        """
        High-level function to get the current screen context.
        """
        img = self.capture_screen()
        if img:
            return self.process_image_for_llm(img)
        return ""
