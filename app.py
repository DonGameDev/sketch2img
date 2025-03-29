# app.py
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import cv2  # Import OpenCV
import time # Import time at the top level
import traceback # For detailed error logging

print("Starting App...")

# --- Configuration ---
# Ensure HF_TOKEN is set as a Secret in your Space settings
HUGGINGFACE_API_KEY = os.environ.get("HF_TOKEN")
HF_MODEL = "stabilityai/stable-diffusion-xl-turbo" # Fast text-to-image model
REQUEST_TIMEOUT = 30 # Seconds to wait for API response
THROTTLE_TIME = 0.75 # Seconds between allowed generation calls (adjust as needed)

if not HUGGINGFACE_API_KEY:
    print("ERROR: Hugging Face API Key (HF_TOKEN) not found. Please set it in Space secrets.")
    # You might want to raise an exception or handle this more gracefully depending on deployment
    # raise ValueError("Hugging Face API Key (HF_TOKEN) not found.")

print(f"Using Model: {HF_MODEL}")
print(f"Throttle Time: {THROTTLE_TIME}s")

# --- Helper Function: Throttling ---
def throttle(wait_time):
    """Decorator that prevents a function from being called more than once every wait_time seconds."""
    last_call_time = 0
    def decorator(func):
        def throttled(*args, **kwargs):
            nonlocal last_call_time
            now = time.time()
            if now - last_call_time > wait_time:
                last_call_time = now
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error during throttled function execution: {e}")
                    traceback.print_exc() # Log detailed error
                    # Decide what to return on error within the throttled function
                    # Returning None might stop UI updates if the output expects an image
                    return None
            else:
                # print("Throttled!") # Uncomment for debugging
                # Return None to prevent updating the output component when throttled
                return None
        return throttled
    return decorator

# --- Image Generation Logic ---
def generate_image_from_drawing(input_np_image):
    """
    Generates an image based on a drawing using SDXL-Turbo.
    Uses basic shape detection on the input drawing to create a text prompt.
    Args:
        input_np_image (np.array | None): NumPy array (H, W, C) from Gradio canvas.
                                           Can be None initially or if canvas is cleared.
    Returns:
        PIL.Image | None: The generated image or None if an error occurs or input is invalid.
    """
    if input_np_image is None:
        print("Input image is None, skipping generation.")
        return None # Return None, Gradio will handle it for the output Image

    # Check if the canvas is effectively empty (e.g., all black or all white)
    # Check for non-zero elements (drawing) or elements not equal to 255 (if background is white)
    if not np.any(input_np_image) or np.all(input_np_image == 255):
         print("Input image is empty, skipping generation.")
         return None

    # --- Dynamic Prompt Engineering (Basic Example) ---
    prompt = "cinematic photo, high detail illustration of " # Base prompt

    try:
        # 1. Basic Shape Detection (Using the input NumPy array directly)
        # Ensure the input array has 3 dimensions (H, W, C) and is RGB
        if input_np_image.ndim != 3 or input_np_image.shape[2] != 3:
             print(f"Warning: Unexpected image shape: {input_np_image.shape}. Skipping shape detection.")
             gray = None # Cannot perform shape detection
        else:
            # Convert RGB to Grayscale for thresholding
            gray = cv2.cvtColor(input_np_image, cv2.COLOR_RGB2GRAY)

        shapes = []
        if gray is not None:
            # Apply thresholding. Assumes dark lines on white background.
            # Use THRESH_BINARY_INV: Pixels below threshold become max value (255), others 0.
            # Adjust threshold value (e.g., 200-240) depending on background color and line intensity.
            # If lines are light on dark bg, use THRESH_BINARY and a lower threshold (e.g., 30-50).
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area to ignore small specks
            min_contour_area = 100 # Adjust as needed based on canvas size
            large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            for contour in large_contours:
                # Approximate the contour shape
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Adjust epsilon factor if needed
                num_vertices = len(approx)
                shape_name = None

                # Basic shape classification
                if num_vertices == 3:
                    shape_name = "triangle"
                elif num_vertices == 4:
                    # Could add logic here to differentiate square/rectangle based on aspect ratio/angles
                    shape_name = "quadrilateral"
                elif num_vertices == 5:
                     shape_name = "pentagon"
                elif num_vertices > 5:
                    # Basic check for circle-like shapes using circularity
                    area = cv2.contourArea(contour)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if 0.75 < circularity < 1.25: # Adjust circularity threshold range
                             shape_name = "circle" # Simplified name
                        else:
                             shape_name = "polygon" # Generic for other complex shapes
                else:
                    shape_name = "line or curve" # If approximation results in < 3 vertices

                # Only add unique shape names found
                if shape_name and shape_name not in shapes:
                    shapes.append(shape_name)

        if shapes:
            prompt += ", ".join(shapes)
        else:
            # Fallback if no shapes detected or analysis skipped
            prompt += "an abstract sketch"

        # Add style modifiers
        prompt += ", trending on artstation, masterpiece, high resolution"

        print(f"Generated Prompt: {prompt}") # Log the prompt being sent

        # --- Hugging Face API Call ---
        if not HUGGINGFACE_API_KEY:
            print("API Key missing, cannot make request.")
            # Maybe return a placeholder indicating the API key issue
            # Or rely on the initial check. For robustness, check again.
            return None

        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "options": {
                "wait_for_model": True, # Important for potentially cold starts
                # SDXL-Turbo often works well with minimal or no negative prompt/guidance
                # "negative_prompt": "blurry, low quality, text, watermark",
                # "guidance_scale": 0.0 # Guidance scale for Turbo models is often 0
            }
        }
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        print(f"API Response Status Code: {response.status_code}")
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # --- Process Response ---
        if response.headers.get("Content-Type") == "image/jpeg" or response.headers.get("Content-Type") == "image/png":
            print("Image received successfully.")
            generated_image_bytes = response.content
            # Convert bytes to PIL Image
            generated_image = Image.open(BytesIO(generated_image_bytes))
            return generated_image # Return PIL Image, Gradio handles it
        else:
            # Handle cases where the API might return JSON with error messages
            print(f"Unexpected content type received: {response.headers.get('Content-Type')}")
            print(f"Response content: {response.text}") # Log the response text
            return None

    except requests.exceptions.Timeout:
        print(f"API Error: Request timed out after {REQUEST_TIMEOUT} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        # Log more details if available (e.g., response content for 4xx/5xx errors)
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Response Content: {e.response.text}")
        return None
    except cv2.error as e:
        print(f"OpenCV Error during shape detection: {e}")
        traceback.print_exc()
        # Proceed without shape info or return None
        # If continuing, ensure 'prompt' has a fallback value
        # For simplicity here, returning None on OpenCV error
        return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_image_from_drawing: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None # Graceful failure

# --- Gradio Interface ---
print("Setting up Gradio interface...")

# Apply throttling to the image generation function
throttled_generate = throttle(wait_time=THROTTLE_TIME)(generate_image_from_drawing)

with gr.Blocks(css="footer {visibility: hidden}") as demo: # Hide default Gradio footer
    gr.Markdown(
        """
        # 🎨 Real-time AI Drawing Canvas
        Draw something below! The app tries to detect basic shapes to create a prompt
        for the SDXL-Turbo model, generating an image in near real-time.
        *(Generation might take a few seconds, especially on first use)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Image(
                label="Draw Here",
                type="numpy",           # Input type is NumPy array
                tool="sketch",          # Use the sketch tool
                image_mode="RGB",       # Ensure input is RGB
                height=512,             # Set canvas height
                width=512,              # Set canvas width
                brush_radius=4          # Adjust brush size
                # interactive=True # Default is True
            )
            gr.Markdown("*(Drawing triggers generation automatically after a short pause)*")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                height=512,
                width=512,
                interactive=False # Output is not interactive
            )
            # Placeholder or status display can be added here if needed

    # --- Event Listener ---
    # Trigger generation when the canvas drawing changes (on mouse release typically)
    canvas.change(
        fn=throttled_generate, # Use the throttled function
        inputs=canvas,
        outputs=output_image,
        # queue=True # Enable queue for smoother handling if generation is slow or multiple users
                     # Might increase perceived latency slightly initially. Test what works best.
    )

    gr.Markdown(f"--- \n*Model: `{HF_MODEL}` | Throttle: `{THROTTLE_TIME}s`*")

print("Gradio Blocks defined.")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.queue() # Enable the queue for better request handling (recommended)
    demo.launch()  # Share=True is not needed when deploying on HF Spaces
    print("Gradio App Launched.")