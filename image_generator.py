# image_generator.py

"""
Backend logic for the Sketch-to-Image Generator application.
Handles image analysis, prompt construction, and API communication.
"""

import os
import time
import asyncio
import requests
import base64
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv

import config

# --- Load Environment Variables ---
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")

if not HUGGINGFACE_API_KEY:
    logging.warning("Hugging Face API Key (HF_TOKEN) not set. Image generation will not work.")

# --- Throttling Mechanism ---
class Throttler:
    """Limits the rate at which a function can be called."""
    def __init__(self, wait_time):
        self.wait_time = wait_time
        self.last_call_time = 0.0

    def throttle(self):
        """Returns True if the call is allowed, False otherwise."""
        now = time.time()
        if now - self.last_call_time > self.wait_time:
            self.last_call_time = now
            return True
        return False

throttler = Throttler(config.THROTTLE_TIME)

# --- Helper Functions ---
def numpy_to_pil(numpy_image_dict):
    """Converts a NumPy array from Gradio's Image input to a PIL Image."""
    if numpy_image_dict is None:
        return None
    try:
        if isinstance(numpy_image_dict, Image.Image):
            return numpy_image_dict.convert("RGB")

        if isinstance(numpy_image_dict, dict):
            numpy_array = None
            for key in ("composite", "image", "background"):
                if numpy_image_dict.get(key) is not None:
                    numpy_array = numpy_image_dict.get(key)
                    break
        else:
            numpy_array = numpy_image_dict

        if isinstance(numpy_array, Image.Image):
            return numpy_array.convert("RGB")

        if numpy_array is None:
            return None

        if len(numpy_array.shape) == 3 and numpy_array.shape[2] in [3, 4]:
            return Image.fromarray(numpy_array[:, :, :3].astype(np.uint8), 'RGB')
        elif len(numpy_array.shape) == 2:
            return Image.fromarray(numpy_array.astype(np.uint8), 'L').convert('RGB')
    except Exception as e:
        logging.error(f"Error converting numpy to PIL: {e}", exc_info=True)
    return None

def pil_to_numpy(pil_image):
    """Converts a PIL Image to a NumPy array."""
    if pil_image is None:
        return None
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    except Exception as e:
        logging.error(f"Error converting PIL to numpy: {e}", exc_info=True)
    return None

def analyze_drawing(pil_image):
    """Analyzes the PIL drawing for shapes and dominant colors."""
    shapes, dominant_colors = [], []
    if pil_image is None:
        return shapes, dominant_colors
    try:
        gray = cv2.cvtColor(pil_to_numpy(pil_image), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > config.MIN_CONTOUR_AREA:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, config.SHAPE_APPROX_EPSILON * perimeter, True)
                shapes.append(_classify_shape(approx))
        dominant_colors = _extract_colors(thresh, pil_to_numpy(pil_image))
    except Exception as e:
        logging.error(f"Error analyzing drawing: {e}", exc_info=True)
    return shapes, dominant_colors

def _classify_shape(approx):
    """Classifies geometric shapes based on their vertices."""
    num_vertices = len(approx)
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "rectangle"
    elif num_vertices > 4:
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        if perimeter == 0:
            return "polygon"
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        return "circle" if config.CIRCLE_CIRCULARITY_THRESHOLD[0] < circularity < config.CIRCLE_CIRCULARITY_THRESHOLD[1] else "polygon"
    return "unknown"

def _extract_colors(thresh, numpy_image_rgb):
    """Extracts dominant colors from the drawing."""
    if numpy_image_rgb is None:
        return []
    try:
        mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) > 0
        pixels = numpy_image_rgb[mask]
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        return [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in unique_colors[sorted_indices[:5]]]
    except Exception as e:
        logging.error(f"Error extracting colors: {e}", exc_info=True)
    return []

# --- Image Generation ---
async def generate_image(pil_image, prompt_override="", selected_model="Stable Diffusion XL Turbo", negative_prompt="", guidance_scale=7.5, seed=-1):
    """Generates an image based on a PIL drawing, text prompts, and advanced parameters."""
    if pil_image is None:
        return None, "Draw something on the canvas."
    if not HUGGINGFACE_API_KEY:
        return None, "API key not set."

    model_id = config.AVAILABLE_MODELS.get(selected_model, list(config.AVAILABLE_MODELS.values())[0])
    prompt, headers, payload = _construct_prompt_and_payload(
        pil_image, prompt_override, negative_prompt, guidance_scale, seed
    )

    try:
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        response = await asyncio.to_thread(
            requests.post, api_url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        if "image/" in response.headers.get("Content-Type", ""):
            return Image.open(BytesIO(response.content)), "Image generated successfully."
    except requests.exceptions.RequestException as e:
        logging.error(f"API Request Error: {e}", exc_info=True)
        # Try to parse a more specific error message from the API response
        error_message = e.response.json().get("error", str(e)) if e.response else str(e)
        return None, f"API Error: {error_message}"
    return None, "Failed to generate image."

def _construct_prompt_and_payload(pil_image, prompt_override, negative_prompt, guidance_scale, seed):
    """Constructs the prompt and API payload for image generation."""
    shapes, colors = analyze_drawing(pil_image)
    prompt = prompt_override.strip() or f"a sketch featuring {', '.join(shapes)} with colors {', '.join(colors)}"

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    parameters = {
        "negative_prompt": negative_prompt.strip() or "low quality, blurry",
        "guidance_scale": guidance_scale,
    }
    # The API expects the seed to be an integer if provided
    if seed != -1:
        parameters["seed"] = seed

    payload = {
        "inputs": prompt,
        "parameters": parameters,
        "options": {"wait_for_model": True}
    }
    return prompt, headers, payload

async def inpaint_image(image_dict, prompt):
    """
    Performs inpainting on an image using a specified mask and prompt.
    The image_dict is expected from Gradio's Image tool, containing 'image' and 'mask'.
    """
    if not all(k in image_dict for k in ["image", "mask"]):
        return None, "Missing image or mask for inpainting."
    if not HUGGINGFACE_API_KEY:
        return None, "API key not set."

    # Convert image and mask to base64
    image = image_dict["image"]
    mask = image_dict["mask"]

    buffered_img = BytesIO()
    image.save(buffered_img, format="PNG")
    img_str = base64.b64encode(buffered_img.getvalue()).decode()

    buffered_mask = BytesIO()
    mask.save(buffered_mask, format="PNG")
    mask_str = base64.b64encode(buffered_mask.getvalue()).decode()

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": img_str,
            "mask_image": mask_str,
        },
        "options": {"wait_for_model": True}
    }

    try:
        api_url = f"https://api-inference.huggingface.co/models/{config.INPAINTING_MODEL}"
        response = await asyncio.to_thread(
            requests.post, api_url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        if "image/" in response.headers.get("Content-Type", ""):
            return Image.open(BytesIO(response.content)), "Inpainting successful."
    except requests.exceptions.RequestException as e:
        logging.error(f"Inpainting API Request Error: {e}", exc_info=True)
        error_message = e.response.json().get("error", str(e)) if e.response else str(e)
        return None, f"API Error: {error_message}"
    return None, "Failed to inpaint image."
