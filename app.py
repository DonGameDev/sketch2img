# app.py
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw # Import ImageDraw as it might be useful for canvas
import numpy as np
import os
import cv2
import time
import traceback
import asyncio

print("Starting Sketch-to-Image App...")

# --- Configuration ---
# Use os.environ.get() with a default to avoid key errors if HF_TOKEN isn't set
HUGGINGFACE_API_KEY = os.environ.get("HF_TOKEN")
# Using a fast model suitable for interactive demos
HF_MODEL = "stabilityai/stable-diffusion-xl-turbo"
REQUEST_TIMEOUT = 45  # Increased timeout slightly for robustness
THROTTLE_TIME = 1.0   # Adjusted throttle time - balance responsiveness and load
# Analysis parameters (tuned slightly)
MIN_CONTOUR_AREA = 75
SHAPE_APPROX_EPSILON = 0.02
CIRCLE_CIRCULARITY_THRESHOLD = (0.65, 1.35) # Wider range

# Check for API key early
if not HUGGINGFACE_API_KEY:
    print("WARNING: Hugging Face API Key (HF_TOKEN) not found. Image generation will not work.")

print(f"Using Model: {HF_MODEL}")
print(f"Throttle Time: {THROTTLE_TIME}s")

# --- State Management / Throttling ---
class Throttler:
    def __init__(self, wait_time):
        self.wait_time = float(wait_time) # Ensure wait_time is float
        self.last_call_time = 0

    def throttle(self):
        now = time.time()
        # Allow initial call immediately
        if self.last_call_time == 0:
             self.last_call_time = now
             return True

        if now - self.last_call_time > self.wait_time:
            self.last_call_time = now
            return True
        return False

throttler = Throttler(THROTTLE_TIME)

# --- Helper Functions ---
def numpy_to_pil(numpy_image):
    """Converts a NumPy array (from gr.Image) to a PIL Image."""
    if numpy_image is not None:
        # Ensure image is in a format PIL can handle (e.g., RGB)
        if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
             return Image.fromarray(numpy_image.astype(np.uint8), 'RGB')
        elif len(numpy_image.shape) == 2: # Handle grayscale if necessary, though canvas is RGB
             return Image.fromarray(numpy_image.astype(np.uint8), 'L').convert('RGB')
        else:
             print(f"Warning: Unexpected numpy image shape: {numpy_image.shape}")
             return None
    return None

def pil_to_numpy(pil_image):
    """Converts a PIL Image to a NumPy array."""
    if pil_image is not None:
        return np.array(pil_image)
    return None

def analyze_drawing(pil_image):
    """Analyzes the PIL drawing for shapes and colors."""
    shapes = []
    dominant_colors = []

    if pil_image is None or not np.any(pil_to_numpy(pil_image)):
        # Handle empty or None images gracefully
        return shapes, dominant_colors

    try:
        # Convert to RGB for consistent processing, then to grayscale for shape detection
        numpy_image_rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_RGB2GRAY)

        # Use a higher threshold or adaptive thresholding if needed
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # Slightly lower threshold

        # Find contours - RETR_EXTERNAL finds outer contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                perimeter = cv2.arcLength(contour, True)
                # Approximate the shape
                approx = cv2.approxPolyDP(contour, SHAPE_APPROX_EPSILON * perimeter, True)
                num_vertices = len(approx)
                shape_name = None

                if num_vertices == 3:
                    shape_name = "triangle"
                elif num_vertices == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.9 <= aspect_ratio <= 1.1: # Tighter aspect ratio for square
                        shape_name = "square"
                    else:
                        shape_name = "rectangle"
                elif num_vertices == 5:
                    shape_name = "pentagon"
                elif num_vertices > 5:
                    # Check for circle
                    area = cv2.contourArea(contour)
                    # Avoid division by zero if perimeter is zero (shouldn't happen with MIN_CONTOUR_AREA > 0)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                    if CIRCLE_CIRCULARITY_THRESHOLD[0] < circularity < CIRCLE_CIRCULARITY_THRESHOLD[1]:
                        shape_name = "circle"
                    else:
                         # Consider simpler description for complex polygons
                         shape_name = "complex shape" # or "polygon"

                elif num_vertices < 3 and perimeter > 10: # Basic detection for lines/curves if area is small
                     shape_name = "line or curve"

                if shape_name and shape_name not in shapes:
                    shapes.append(shape_name)

        # Basic color analysis - get dominant colors from the drawing itself (not the background)
        # Mask the image with the threshold to get only drawn pixels
        masked_image_np = cv2.bitwise_and(numpy_image_rgb, numpy_image_rgb, mask=thresh)
        # Convert back to PIL for color analysis
        masked_pil_image = Image.fromarray(masked_image_np)

        # Get colors, excluding the background color (black due to masking)
        # maxcolors=256 to get a decent sample, then filter
        colors_with_counts = masked_pil_image.getcolors(maxcolors=256)
        if colors_with_counts:
            # Sort by count (most frequent first), exclude black (0,0,0)
            sorted_colors = sorted([c for c in colors_with_counts if c[1] != (0, 0, 0)], key=lambda x: x[0], reverse=True)
            # Take top N colors (e.g., top 3-5)
            top_n = 5
            for count, color in sorted_colors[:top_n]:
                 # Exclude near-white drawn lines if any slip through thresholding
                 if not (240 < color[0] <= 255 and 240 < color[1] <= 255 and 240 < color[2] <= 255):
                    dominant_colors.append(f"#{''.join(f'{c:02x}' for c in color[:3])}") # Hex code

    except cv2.Error as e:
        print(f"OpenCV Error during analysis: {e}")
        traceback.print_exc()
        # Continue, returning empty lists
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        traceback.print_exc()
        # Continue, returning empty lists

    return shapes, dominant_colors

# --- Image Generation Logic (Asynchronous) ---
async def generate_image_from_drawing(pil_image, prompt_override=""):
    """
    Generates an image based on a PIL drawing and optional text prompt.
    Returns a PIL Image or a tuple (None, error_message_string).
    """
    if pil_image is None or not np.any(pil_to_numpy(pil_image)):
        print("Input image is empty/None, skipping generation.")
        return None, "No drawing detected for generation."

    if not HUGGINGFACE_API_KEY:
        return None, "Error: Hugging Face API Key not configured."

    # --- Prompt Construction ---
    prompt = "cinematic photo, high detail illustration of " # Base prompt
    detected_shapes, dominant_colors = analyze_drawing(pil_image)

    if detected_shapes:
        prompt += ", ".join(detected_shapes)
    else:
        prompt += "an abstract sketch" # Default if no shapes detected

    if dominant_colors:
        prompt += f", with colors: {', '.join(dominant_colors)}"

    if prompt_override:
        # Combine override with detected elements
        prompt = f"{prompt_override}, based on a drawing of {', '.join(detected_shapes) or 'an abstract form'}"
        if dominant_colors:
             prompt += f" with colors: {', '.join(dominant_colors)}"

    # Add negative prompt elements to avoid common issues
    negative_prompt = "low quality, blurry, distortion, ugly, disfigured, poor anatomy, wrong proportions"

    prompt += ", trending on artstation, masterpiece, high resolution" # Quality enhancers
    print(f"Generated Prompt: {prompt}")

    # --- API Request ---
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": { # Use parameters key for model specific settings if needed (optional)
             "negative_prompt": negative_prompt
        },
        "options": {"wait_for_model": True} # Wait for the model to load if it's not active
    }
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    try:
        print(f"Sending request to {api_url}...")
        # Use asyncio.to_thread to run the synchronous requests.post call in a separate thread
        response = await asyncio.to_thread(requests.post, api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        print(f"API Response Status Code: {response.status_code}")

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Check content type to ensure it's an image
        content_type = response.headers.get("Content-Type")
        if content_type and ("image/jpeg" in content_type or "image/png" in content_type):
            generated_image_bytes = response.content
            generated_image = Image.open(BytesIO(generated_image_bytes))
            return generated_image, "Generation complete."
        else:
            # Handle cases where the API returns an error message in the response body
            error_message = response.text
            print(f"API returned non-image content (Status: {response.status_code}, Content-Type: {content_type}): {error_message}")
            # Attempt to parse common HF API error structures if possible
            try:
                 error_data = response.json()
                 if 'error' in error_data:
                      error_message = error_data['error']
            except json.JSONDecodeError:
                 pass # response.text is used if json parsing fails
            return None, f"Error from API: {error_message}"

    except requests.exceptions.Timeout:
        print(f"API Timeout after {REQUEST_TIMEOUT} seconds.")
        return None, f"Error: API request timed out after {REQUEST_TIMEOUT} seconds."
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        # Include response text in error message if available
        error_text = getattr(e.response, 'text', 'No response text')
        print(f"Response Text: {error_text}")
        return None, f"Error during API request: {e}. Response: {error_text[:200]}..." # Truncate long responses
    except Exception as e:
        print(f"Unexpected Error during generation: {e}")
        traceback.print_exc()
        return None, f"An unexpected error occurred: {e}"

# --- Gradio Interface Logic ---
print("Setting up Gradio interface...")

async def process_drawing(canvas_data, current_prompt):
    """
    Processes the drawing from the canvas and triggers image generation,
    applying throttling.
    """
    # canvas_data from gr.Image(type="numpy") is a numpy array or None

    # 1. Check if drawing data exists and is not completely blank
    if canvas_data is None or not np.any(canvas_data):
        print("Canvas data is None or empty.")
        return None, "Draw something to get started!"

    # Convert numpy array to PIL image for further processing
    pil_image = numpy_to_pil(canvas_data)
    if pil_image is None:
         return None, "Error processing drawing data."

    # 2. Apply throttling
    if not throttler.throttle():
        # Return the *previous* output image if available, and a throttled status
        # This requires managing the last generated image state.
        # For simplicity, we'll just return None and the status message,
        # meaning the output image will clear or remain unchanged from the last successful gen.
        # A more advanced approach would involve storing the last successful output.
        return None, f"Throttled: Please wait {THROTTLE_TIME}s between updates."

    print("Processing drawing...")
    # 3. Trigger asynchronous image generation
    # generate_image_from_drawing now returns (PIL_Image or None, status_string)
    generated_image, status_message = await generate_image_from_drawing(pil_image, current_prompt)

    # 4. Return results to Gradio outputs
    # generated_image will be either a PIL Image object or None if an error occurred
    return generated_image, status_message

def clear_canvas():
    """Clears the drawing canvas."""
    # Returning None for a gr.Image input clears it
    return None, "Canvas cleared." # Also clear status

# --- Gradio Blocks Definition ---
# Using Blocks for more layout control
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🎨 Real-time Sketch to Image AI Generator
        Draw on the canvas and watch AI generate an image based on your sketch!
        *(Generation happens every few seconds as you draw, thanks to throttling. Empty drawings are ignored.)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Input drawing canvas
            canvas = gr.Image(
                label="Draw Here",
                type="numpy", # Receive drawing as a numpy array
                image_mode="RGB", # Expect RGB data
                height=512,
                width=512,
                interactive=True, # Allow drawing
            )
            with gr.Row():
                # Button to clear the canvas
                clear_button = gr.Button("Clear Canvas")
                # Note: A true "Undo" needs history state management, which is complex
                # for real-time drawing updates. Keeping it simple for now.

            # Textbox for optional additional prompt
            prompt_input = gr.Textbox(
                label="Optional Text Prompt",
                placeholder="e.g., 'a red apple on a table', 'a robot cat'",
                lines=2 # Allow multiple lines
            )

            # Display detected elements (shapes/colors)
            prompt_analysis_display = gr.Textbox(
                label="Detected Sketch Elements",
                interactive=False, # Not editable by user
                lines=3 # Show multiple lines of info
            )

        with gr.Column(scale=1):
            # Output generated image display
            output_image = gr.Image(
                label="Generated Image",
                height=512,
                width=512,
                interactive=False # Display only
            )
            # Status message display
            status_display = gr.Textbox(
                label="Status",
                interactive=False, # Not editable by user
                lines=2 # Show multiple lines of status
            )

    # --- Event Listeners ---

    # Trigger image generation when the canvas is edited (drawn on) or text prompt changes
    # Use gr.State to potentially hold last successful outputs if needed for throttling "return previous"
    # but let's keep it simple and just return None on throttle for this version.
    canvas.change(
        fn=process_drawing, # The async function to call
        inputs=[canvas, prompt_input],
        outputs=[output_image, status_display],
        # queue=True is important for async functions and handling multiple rapid triggers
        queue=True
    )
    # Also trigger generation if the text prompt is submitted (e.g., press Enter or unfocus)
    prompt_input.submit(
        fn=process_drawing,
        inputs=[canvas, prompt_input],
        outputs=[output_image, status_display],
        queue=True
    )


    # Clear button action
    clear_button.click(
        fn=clear_canvas,
        inputs=[],
        outputs=[canvas, status_display], # Clear both canvas and status message
        queue=False # Clearing is fast, no need to queue
    )


    gr.Markdown(f"--- \n*Model: `{HF_MODEL}` | Throttle: `{THROTTLE_TIME}s`* \n*Remember to set your HF_TOKEN environment variable.*")

    # --- Analysis Display Update (Faster, doesn't wait for generation) ---
    # This lambda updates the analysis display *before* the main generation process runs
    # Uses queue=False so it's more responsive
    def update_analysis_display(canvas_data):
        if canvas_data is None:
            return "No drawing yet."
        pil_img = numpy_to_pil(canvas_data)
        if pil_img:
            shapes, colors = analyze_drawing(pil_img)
            shape_text = f"Shapes: {', '.join(shapes) or 'None'}"
            color_text = f"Colors: {', '.join(colors) or 'None'}"
            return f"{shape_text}\n{color_text}"
        return "Error analyzing drawing."

    canvas.change(
        fn=update_analysis_display,
        inputs=[canvas],
        outputs=[prompt_analysis_display],
        queue=False # Process this update immediately without waiting for the main queue
    )


print("Gradio Blocks defined.")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # .queue() enables the queueing system defined in event listeners
    demo.queue()
    # demo.launch() starts the web server
    # Set debug=True during development for more detailed error messages
    demo.launch()
    print("Gradio App Launched.")
    