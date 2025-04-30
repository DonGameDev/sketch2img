# app.py
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2  # Import OpenCV
import time
import traceback
import asyncio # For asynchronous tasks

print("Starting Enhanced App...")

# --- Configuration ---
HUGGINGFACE_API_KEY = os.environ.get("HF_TOKEN")
HF_MODEL = "stabilityai/stable-diffusion-xl-turbo" # Fast text-to-image model
REQUEST_TIMEOUT = 30 # Seconds
THROTTLE_TIME = 0.5 # Reduced slightly with potential async
MIN_CONTOUR_AREA = 50 # Adjusted min contour area
SHAPE_APPROX_EPSILON = 0.01 # Tuned shape approximation
CIRCLE_CIRCULARITY_THRESHOLD = (0.7, 1.3) # Wider range for circles

if not HUGGINGFACE_API_KEY:
    print("ERROR: Hugging Face API Key (HF_TOKEN) not found.")

print(f"Using Model: {HF_MODEL}")
print(f"Throttle Time: {THROTTLE_TIME}s")

# --- State Management ---
class DrawingState:
    def __init__(self):
        self.last_call_time = 0
        self.drawn_image = None # To store the PIL image of the drawing

drawing_state = DrawingState()

# --- Helper Functions ---
class Throttler:
    def __init__(self, wait_time):
        self.wait_time = wait_time
        self.last_call_time = 0

    def throttle(self):
        now = time.time()
        if now - self.last_call_time > self.wait_time:
            self.last_call_time = now
            return True
        return False

throttler = Throttler(THROTTLE_TIME)

def numpy_to_pil(numpy_image):
    """Converts a NumPy array to a PIL Image."""
    if numpy_image is not None:
        return Image.fromarray(numpy_image.astype(np.uint8))
    return None

def pil_to_numpy(pil_image):
    """Converts a PIL Image to a NumPy array."""
    if pil_image is not None:
        return np.array(pil_image)
    return None

def analyze_drawing(pil_image):
    """Analyzes the PIL drawing for shapes and colors."""
    if pil_image is None:
        return [], []

    numpy_image = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, SHAPE_APPROX_EPSILON * perimeter, True)
            num_vertices = len(approx)
            shape_name = None

            if num_vertices == 3:
                shape_name = "triangle"
            elif num_vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = "square"
                else:
                    shape_name = "rectangle"
            elif num_vertices == 5:
                shape_name = "pentagon"
            elif num_vertices > 5:
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                if CIRCLE_CIRCULARITY_THRESHOLD[0] < circularity < CIRCLE_CIRCULARITY_THRESHOLD[1]:
                    shape_name = "circle"
                else:
                    shape_name = "polygon"
            elif num_vertices < 3 and perimeter > 20: # Basic line/curve detection
                shape_name = "line or curve"

            if shape_name and shape_name not in shapes:
                shapes.append(shape_name)

    # Basic color analysis (dominant colors)
    colors = pil_image.getcolors(maxcolors=5) # Get up to 5 dominant colors
    dominant_colors = []
    if colors:
        for count, color in colors:
            # Exclude near-white or near-black as "dominant" for simplicity
            if not (200 < color[0] < 256 and 200 < color[1] < 256 and 200 < color[2] < 256) and \
               not (color[0] < 50 and color[1] < 50 and color[2] < 50):
                dominant_colors.append(f"#{''.join(f'{c:02x}' for c in color[:3])}") # Hex code

    return shapes, dominant_colors

# --- Image Generation Logic (Asynchronous) ---
async def generate_image_from_drawing(input_pil_image, prompt_override=""):
    """Generates an image based on a PIL drawing and optional text prompt."""
    if input_pil_image is None:
        print("Input image is None, skipping generation.")
        return None

    if not np.any(pil_to_numpy(input_pil_image)) or np.all(pil_to_numpy(input_pil_image) == 255):
        print("Input image is empty, skipping generation.")
        return None

    prompt = "cinematic photo, high detail illustration of " # Base prompt
    detected_shapes, dominant_colors = analyze_drawing(input_pil_image)

    if detected_shapes:
        prompt += ", ".join(detected_shapes)
    else:
        prompt += "an abstract sketch"

    if dominant_colors:
        prompt += f", with colors: {', '.join(dominant_colors)}"

    if prompt_override:
        prompt = f"{prompt_override}, based on a drawing of {', '.join(detected_shapes) or 'an abstract form'}"

    prompt += ", trending on artstation, masterpiece, high resolution"
    print(f"Generated Prompt: {prompt}")

    if not HUGGINGFACE_API_KEY:
        print("API Key missing, cannot make request.")
        return "Error: API Key not configured."

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    try:
        print(f"Sending request to {api_url}...")
        response = await asyncio.to_thread(requests.post, api_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        print(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()

        if response.headers.get("Content-Type") in ["image/jpeg", "image/png"]:
            generated_image_bytes = response.content
            generated_image = Image.open(BytesIO(generated_image_bytes))
            return generated_image
        else:
            print(f"Unexpected content type: {response.headers.get('Content-Type')}, Response: {response.text}")
            return "Error: Unexpected API response."

    except requests.exceptions.Timeout:
        print(f"API Timeout after {REQUEST_TIMEOUT} seconds.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}, Response: {getattr(e.response, 'text', '')}")
        return f"Error: API request failed ({e})."
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        traceback.print_exc()
        return "Error: Drawing analysis failed."
    except Exception as e:
        print(f"Unexpected Error: {e}")
        traceback.print_exc()
        return "Error: Image generation failed."

# --- Gradio Interface ---
print("Setting up Enhanced Gradio interface...")

async def process_drawing(canvas_data, current_prompt):
    """Processes the drawing and generates the image, applying throttling."""
    if not throttler.throttle():
        return None, "Throttled: Please wait a moment."

    if canvas_data is None:
        return None, "No drawing input."

    pil_image = numpy_to_pil(canvas_data)
    drawing_state.drawn_image = pil_image # Store the current drawing

    if pil_image:
        generated_image = await generate_image_from_drawing(pil_image, current_prompt)
        return generated_image, "Generation complete."
    else:
        return None, "Invalid drawing input."

def clear_canvas():
    """Clears the drawing canvas."""
    return None

def undo_drawing(previous_images):
    """Undoes the last drawing action (basic implementation)."""
    if previous_images:
        return previous_images[-1] # Returns the last saved state
    return None

# --- Gradio Blocks ---
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🎨 Enhanced Real-time AI Drawing Canvas
        Draw, add text, and watch AI generate images based on your creations!
        *(Generation might take a few seconds)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Image(
                label="Draw Here",
                source="canvas",  # Explicitly set the source to 'canvas'
                type="numpy",
                image_mode="RGB",
                height=512,
                width=512,
                interactive=True
                # We'll omit brush_radius and tool_kwargs for now             
                        )
            with gr.Row():
                clear_button = gr.Button("Clear Canvas")
                # Basic "undo" by just clearing - more sophisticated undo would require state tracking
                undo_button = gr.Button("Undo (Clear)")
            prompt_input = gr.Textbox(label="Optional Text Prompt", placeholder="Add extra details to influence the image")
            prompt_display = gr.Textbox(label="Detected Shapes & Colors", interactive=False)

            def update_prompt_display(pil_image):
                if pil_image:
                    shapes, colors = analyze_drawing(pil_image)
                    prompt_info = f"Detected Shapes: {', '.join(shapes) or 'None'}\nDetected Colors: {', '.join(colors) or 'None'}"
                    return prompt_info
                return "No drawing to analyze."

            canvas.edit(fn=lambda x: update_prompt_display(numpy_to_pil(x)), inputs=canvas, outputs=prompt_display, queue=False)

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                height=512,
                width=512,
                interactive=False
            )
            status_display = gr.Textbox(label="Status", interactive=False)

    # --- Event Listeners ---
    canvas.edit(
        fn=process_drawing,
        inputs=[canvas, prompt_input],
        outputs=[output_image, status_display],
        queue=True
    )
    clear_button.click(inputs=[], outputs=canvas, fn=clear_canvas)
    undo_button.click(inputs=[], outputs=canvas, fn=clear_canvas) # Basic undo as clear

    gr.Markdown(f"--- \n*Model: `{HF_MODEL}` | Throttle: `{THROTTLE_TIME}s`*")

print("Enhanced Gradio Blocks defined.")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Enhanced Gradio App...")
    demo.queue()
    demo.launch()
    print("Enhanced Gradio App Launched.")
    