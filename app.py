# app.py

"""
Sketch-to-Image Generator (fixed and refactored core helpers)
This application generates images based on user sketches using the Hugging Face API
or a local pipeline when available.
"""

import os
import time
import traceback
import asyncio
import requests
from io import BytesIO
from math import pi
from PIL import Image
import numpy as np
import cv2
import gradio as gr
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")

# --- Configuration ---
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-turbo")
REQUEST_TIMEOUT = 60
THROTTLE_TIME = 1.5
MIN_CONTOUR_AREA = 100
SHAPE_APPROX_EPSILON = 0.02
CIRCLE_CIRCULARITY_THRESHOLD = (0.6, 1.4)

if not HUGGINGFACE_API_KEY:
    print("WARNING: Hugging Face API Key (HF_TOKEN) not set. Image generation will not work in API mode.")

# --- Throttling Mechanism ---
class Throttler:
    def __init__(self, wait_time):
        self.wait_time = wait_time
        self.last_call_time = 0.0

    def throttle(self):
        now = time.time()
        if now - self.last_call_time > self.wait_time:
            self.last_call_time = now
            return True
        return False

throttler = Throttler(THROTTLE_TIME)

# --- Helper Functions ---
def numpy_to_pil(numpy_image_dict):
    if numpy_image_dict is None:
        return None
    try:
        numpy_array = numpy_image_dict.get('image') if isinstance(numpy_image_dict, dict) else numpy_image_dict
        if len(numpy_array.shape) == 3 and numpy_array.shape[2] in [3, 4]:
            return Image.fromarray(numpy_array[:, :, :3].astype(np.uint8), 'RGB')
        elif len(numpy_array.shape) == 2:
            return Image.fromarray(numpy_array.astype(np.uint8), 'L').convert('RGB')
    except Exception as e:
        print(f"Error converting numpy to PIL: {e}")
    return None


def pil_to_numpy(pil_image):
    if pil_image is None:
        return None
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    except Exception as e:
        print(f"Error converting PIL to numpy: {e}")
    return None


def analyze_drawing(pil_image):
    """Analyze drawing: find shapes and dominant colors."""
    shapes, dominant_colors = [], []
    if pil_image is None:
        return shapes, dominant_colors
    try:
        numpy_rgb = pil_to_numpy(pil_image)
        gray = cv2.cvtColor(numpy_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, SHAPE_APPROX_EPSILON * perimeter, True)
                circularity = (4 * pi * cv2.contourArea(contour) / (perimeter * perimeter)) if perimeter > 0 else 0
                shapes.append(_classify_shape(approx, circularity))
        dominant_colors = _extract_colors(thresh, numpy_rgb)
    except Exception as e:
        print(f"Error analyzing drawing: {e}")
    return shapes, dominant_colors


def _classify_shape(approx, circularity=0.0):
    num_vertices = len(approx)
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "rectangle"
    elif num_vertices > 4:
        return "circle" if CIRCLE_CIRCULARITY_THRESHOLD[0] < circularity < CIRCLE_CIRCULARITY_THRESHOLD[1] else "polygon"
    return "unknown"


def _extract_colors(thresh, numpy_image_rgb):
    if numpy_image_rgb is None:
        return []
    try:
        mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) > 0
        if mask.shape[:2] != numpy_image_rgb.shape[:2]:
            return []
        pixels = numpy_image_rgb[mask]
        if pixels.size == 0:
            return []
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        return [f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}" for color in unique_colors[sorted_indices[:5]]]
    except Exception as e:
        print(f"Error extracting colors: {e}")
    return []

# --- Image Generation (Hugging Face Inference API fallback) ---
async def generate_image_from_drawing(pil_image, prompt_override=""):
    if pil_image is None or not pil_to_numpy(pil_image).any():
        return None, "Draw something on the canvas."
    # If throttled, do not spam calls
    if not throttler.throttle():
        return None, "Throttled: waiting for next allowed call"
    prompt, headers, payload = _construct_prompt_and_payload(pil_image, prompt_override)
    if not HUGGINGFACE_API_KEY:
        return None, "API key not set."
    try:
        response = await asyncio.to_thread(
            requests.post,
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        if "image/" in response.headers.get("Content-Type", ""):
            return Image.open(BytesIO(response.content)), "Image generated successfully."
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
    return None, "Failed to generate image."


def _construct_prompt_and_payload(pil_image, prompt_override):
    shapes, colors = analyze_drawing(pil_image)
    prompt = prompt_override.strip() or f"a sketch featuring {', '.join(shapes) if shapes else 'various shapes'} with colors {', '.join(colors) if colors else 'unknown'}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"negative_prompt": "low quality, blurry", "guidance_scale": 7.5},
        "options": {"wait_for_model": True}
    }
    return prompt, headers, payload

# --- Gradio Interface (simple local demo) ---
with gr.Blocks() as demo:
    gr.Markdown("# Sketch-to-Image Generator (Local/Hub mode)")
    canvas = gr.Image(label="Draw Here", type="numpy", image_mode="RGB", height=512, width=512, interactive=True)
    prompt_input = gr.Textbox(label="Text Prompt", placeholder="Optional description")
    output_image = gr.Image(label="Generated Image", interactive=False)
    status_display = gr.Textbox(label="Status", interactive=False)
    canvas.change(fn=generate_image_from_drawing, inputs=[canvas, prompt_input], outputs=[output_image, status_display])

if __name__ == "__main__":
    demo.launch(share=True)
