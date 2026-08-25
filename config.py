# config.py

"""
Configuration settings for the Sketch-to-Image Generator application.
"""

# --- API and Model Configuration ---
AVAILABLE_MODELS = {
    "Stable Diffusion XL Turbo": "stabilityai/stable-diffusion-xl-turbo",
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
    # Using a model specifically fine-tuned for inpainting
    "Stable Diffusion 2 Inpainting": "stabilityai/stable-diffusion-2-inpainting",
}
INPAINTING_MODEL = "stabilityai/stable-diffusion-2-inpainting"

# --- Application Behavior ---
REQUEST_TIMEOUT = 90  # Increased timeout for potentially slower models
THROTTLE_TIME = 1.5   # seconds

# --- Image Analysis Parameters ---
MIN_CONTOUR_AREA = 100
SHAPE_APPROX_EPSILON = 0.02
CIRCLE_CIRCULARITY_THRESHOLD = (0.6, 1.4)
