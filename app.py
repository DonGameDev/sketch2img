# app.py
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import cv2
import time
import traceback
import asyncio # For asynchronous tasks

print("Starting Sketch-to-Image App...")

# --- Configuration ---
# Get API Key from environment variables - essential for Hugging Face Spaces secrets
HUGGINGFACE_API_KEY = os.environ.get("HF_TOKEN")
# Fast text-to-image model suitable for interactive demos
HF_MODEL = "stabilityai/stable-diffusion-xl-turbo"
REQUEST_TIMEOUT = 60 # Increased timeout for generation
THROTTLE_TIME = 1.5  # Time in seconds to wait between API calls triggered by drawing
# Analysis parameters (tuned slightly)
MIN_CONTOUR_AREA = 100 # Increased min contour area
SHAPE_APPROX_EPSILON = 0.02
CIRCLE_CIRCULARITY_THRESHOLD = (0.6, 1.4) # Wider range for circles

# Check if API key is set
if not HUGGINGFACE_API_KEY:
    print("WARNING: Hugging Face API Key (HF_TOKEN) environment variable not set.")
    print("Image generation will not work without the API key.")

print(f"Using Model: {HF_MODEL}")
print(f"Throttle Time: {THROTTLE_TIME}s")

# --- Throttling Mechanism ---
class Throttler:
    """Limits the rate at which a function can be called."""
    def __init__(self, wait_time):
        self.wait_time = float(wait_time)
        self.last_call_time = 0.0 # Use float for time

    def throttle(self):
        """Returns True if call is allowed, False otherwise."""
        now = time.time()
        # Allow the very first call
        if self.last_call_time == 0.0:
             self.last_call_time = now
             return True

        if now - self.last_call_time > self.wait_time:
            self.last_call_time = now
            return True
        return False

throttler = Throttler(THROTTLE_TIME)

# --- Helper Functions ---
def numpy_to_pil(numpy_image_dict):
    """
    Converts a NumPy array from Gradio's Image input (interactive=True)
    to a PIL Image. Handles dictionary format from recent Gradio versions.
    """
    if numpy_image_dict is None:
        return None

    # Gradio 4.x interactive image returns a dict like {'mask': None, 'image': array}
    if isinstance(numpy_image_dict, dict) and 'image' in numpy_image_dict:
        numpy_array = numpy_image_dict['image']
    elif isinstance(numpy_image_dict, np.ndarray):
        # Handle older Gradio versions or different output types if needed
        numpy_array = numpy_image_dict
    else:
        print(f"Warning: Unexpected input type for numpy_to_pil: {type(numpy_image_dict)}")
        return None

    if numpy_array is None:
        return None

    try:
        # Ensure array is in a format PIL can handle (e.g., RGB)
        if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
             # If it's already RGB, just convert
             return Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
        elif len(numpy_array.shape) == 2:
             # If grayscale, convert to RGB
             return Image.fromarray(numpy_array.astype(np.uint8), 'L').convert('RGB')
        elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 4:
             # If RGBA, convert to RGB (discarding alpha)
             return Image.fromarray(numpy_array[:,:,:3].astype(np.uint8), 'RGB')
        else:
             print(f"Warning: Cannot convert numpy array with shape {numpy_array.shape} to PIL.")
             return None
    except Exception as e:
         print(f"Error converting numpy to PIL: {e}")
         traceback.print_exc()
         return None


def pil_to_numpy(pil_image):
    """Converts a PIL Image to a NumPy array."""
    if pil_image is None:
        return None
    try:
        # Ensure PIL image is in RGB mode before converting to numpy
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    except Exception as e:
        print(f"Error converting PIL to numpy: {e}")
        traceback.print_exc()
        return None

def analyze_drawing(pil_image):
    """Analyzes the PIL drawing for shapes and colors."""
    shapes = []
    dominant_colors = []

    if pil_image is None:
        return shapes, dominant_colors # Return empty for None input

    # Convert to grayscale and apply inverse binary threshold
    # A higher threshold (e.g., 220) works well for distinguishing black ink from white background
    try:
        numpy_image_rgb = pil_to_numpy(pil_image)
        if numpy_image_rgb is None:
            return shapes, dominant_colors # Handle conversion failure

        gray = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_RGB2GRAY)
        # Use a threshold that isolates the drawing (assuming dark on light)
        # Pixels darker than thresh_val become 255, others 0. Inverted.
        thresh_val = 220
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter by area to ignore small dots/noise
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: # Avoid division by zero
                    continue

                approx = cv2.approxPolyDP(contour, SHAPE_APPROX_EPSILON * perimeter, True)
                num_vertices = len(approx)
                shape_name = None

                if num_vertices == 3:
                    shape_name = "triangle"
                elif num_vertices == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    # Check for square vs rectangle based on aspect ratio
                    if 0.9 <= aspect_ratio <= 1.1:
                        shape_name = "square"
                    else:
                        shape_name = "rectangle"
                elif num_vertices == 5:
                    shape_name = "pentagon"
                elif num_vertices > 5:
                    area = cv2.contourArea(contour)
                    # Calculate circularity
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    # Check if it's close to a circle (circularity ~ 1)
                    if CIRCLE_CIRCULARITY_THRESHOLD[0] < circularity < CIRCLE_CIRCULARITY_THRESHOLD[1]:
                        shape_name = "circle"
                    else:
                        shape_name = "polygon" # General polygon for > 5 vertices not circle-like
                elif num_vertices < 3 and perimeter > 20: # Basic line/curve detection for small number of points but some length
                     shape_name = "line or curve"


                # Add unique shape names
                if shape_name and shape_name not in shapes:
                    shapes.append(shape_name)

        # Basic color analysis on the drawn parts
        # Create a mask from the thresholded image to isolate drawn pixels
        drawn_pixels_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) > 0 # Convert mask to 3 channels for RGB comparison
        # Apply the mask to the original RGB image
        drawn_colors_np = numpy_image_rgb[drawn_pixels_mask]

        if drawn_colors_np.size > 0:
            # Reshape to a list of pixels (N, 3)
            pixels = drawn_colors_np.reshape(-1, 3)
            # Find unique colors and their counts
            # Consider simplifying colors if too many unique ones (e.g., k-means clustering)
            # For now, let's find the most frequent *distinct* colors, ignoring near-white background that might remain
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # Sort by count descending and filter out very light colors (near white)
            sorted_indices = np.argsort(counts)[::-1]
            top_n_colors = 0
            for i in sorted_indices:
                 color = unique_colors[i]
                 # Check if the color is NOT close to white
                 if not (color[0] > 240 and color[1] > 240 and color[2] > 240):
                      dominant_colors.append(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                      top_n_colors += 1
                      if top_n_colors >= 5: # Limit to top 5 dominant colors
                           break

    except cv2.Error as e:
        print(f"OpenCV Error during analysis: {e}")
        traceback.print_exc()
        # Return empty lists on error
        return [], []
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        traceback.print_exc()
        # Return empty lists on error
        return [], []

    return shapes, dominant_colors

# --- Image Generation Logic (Asynchronous) ---
async def generate_image_from_drawing(pil_image, prompt_override=""):
    """
    Generates an image based on a PIL drawing and optional text prompt using HF API.
    Returns a tuple: (PIL_Image or None, status_message_string).
    """
    # Check for empty/None input image before proceeding
    if pil_image is None or not np.any(pil_to_numpy(pil_image)):
        print("Input image is empty/None, skipping generation API call.")
        # Return None for image and an informative status message
        return None, "Draw something on the canvas to generate an image!"

    # Check if API key is available
    if not HUGGINGFACE_API_KEY:
        return None, "Error: Hugging Face API Key (HF_TOKEN) not configured."

    # --- Prompt Construction ---
    # Start with a base prompt for image style
    prompt = "cinematic photo, high detail, professional illustration of "
    detected_shapes, dominant_colors = analyze_drawing(pil_image)

    # Add detected elements to the prompt
    if detected_shapes:
        prompt += ", ".join(detected_shapes)
    else:
        # If no shapes detected (e.g., abstract scribbles), use a default description
        prompt += "an abstract drawing"

    # Add colors to the prompt if detected
    if dominant_colors:
        prompt += f", featuring colors: {', '.join(dominant_colors)}"

    # If an override prompt is provided, use it but reference the drawing
    if prompt_override.strip(): # Use strip() to check if override is not just whitespace
        prompt = f"{prompt_override.strip()}, based on a drawing of {', '.join(detected_shapes) or 'an abstract form'}"
        # Optionally add colors even with override if they were detected
        if dominant_colors:
             prompt += f" with colors: {', '.join(dominant_colors)}"

    # Add quality and style enhancers to the prompt
    prompt += ", trending on artstation, masterpiece, high resolution, 4k"

    # Define a negative prompt to avoid undesirable features
    negative_prompt = "low quality, blurry, distorted, text, signature, watermark, poorly drawn, bad anatomy, fused fingers, extra limbs"

    print(f"Generated Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")

    # --- API Request ---
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
             "negative_prompt": negative_prompt,
             # Add other parameters if the model supports them, e.g., height, width, steps
             # Note: Turbo models usually have fixed steps/performance settings
             "guidance_scale": 0 # SDXL-Turbo often works best with guidance_scale 0 or low
        },
        "options": {
            "wait_for_model": True # Wait for the model to load if it's cold
        }
    }
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    try:
        print(f"Sending request to {api_url}...")
        # Use asyncio.to_thread to run the synchronous requests call without blocking the event loop
        response = await asyncio.to_thread(
            requests.post,
            api_url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT # Apply the timeout
        )
        print(f"API Response Status Code: {response.status_code}")

        # Raise an exception for HTTP errors (4xx or 5xx)
        response.raise_for_status()

        # Check if the response is an image
        content_type = response.headers.get("Content-Type", "")
        if "image/" in content_type:
            generated_image_bytes = response.content
            generated_image = Image.open(BytesIO(generated_image_bytes))
            print("Image generated successfully.")
            # Return the PIL image and a success status
            return generated_image, "Generation complete."
        else:
            # If API returned something other than an image (e.g., error message as JSON or text)
            error_message = response.text
            print(f"API returned non-image content (Status: {response.status_code}, Content-Type: {content_type}): {error_message[:500]}...") # Print part of response
            # Attempt to parse JSON error message if applicable
            try:
                 error_data = response.json()
                 if isinstance(error_data, dict) and 'error' in error_data:
                      error_message = error_data.get('error', 'Unknown API error')
                 elif isinstance(error_data, list) and error_data and 'error' in error_data[0]:
                       error_message = error_data[0].get('error', 'Unknown API error')
            except json.JSONDecodeError:
                 pass # If not JSON, use the text response

            # Return None for the image and the error message in the status
            return None, f"Error from API: {error_message}"

    except requests.exceptions.Timeout:
        print(f"API Timeout after {REQUEST_TIMEOUT} seconds.")
        return None, f"Error: API request timed out after {REQUEST_TIMEOUT} seconds."
    except requests.exceptions.RequestException as e:
        # Handle other requests exceptions (connection errors, etc.)
        print(f"API Request Error: {e}")
        # Include response text in error message if available
        error_text = getattr(e.response, 'text', 'N/A')
        print(f"Response Text: {error_text[:500]}...")
        return None, f"Error during API request: {e}. Response: {error_text[:200]}..."
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred during generation: {e}")
        traceback.print_exc()
        return None, f"An unexpected error occurred: {e}"

# --- Gradio Interface Logic ---
print("Setting up Gradio interface...")

async def process_drawing_and_generate(canvas_input_data, current_prompt):
    """
    Processes the drawing input from Gradio, applies throttling, and triggers
    the asynchronous image generation.
    """
    # canvas_input_data will be a dict from gr.Image(interactive=True, type='numpy') in Gradio 4+

    # 1. Basic check if drawing data exists
    if canvas_input_data is None or not canvas_input_data.get('image', None) is not None:
        # Return None for output image and an initial status message
        return None, "Draw something on the canvas to start!"

    # Convert the input data (dict containing numpy array) to a PIL image
    pil_image = numpy_to_pil(canvas_input_data)
    if pil_image is None:
         # Handle case where numpy_to_pil failed
         return None, "Error processing drawing data."

    # Check if the drawing is essentially blank white (or transparent if mask was present)
    # We convert PIL back to numpy temporarily for this check
    if pil_image is not None and np.all(pil_to_numpy(pil_image) == 255):
         # Image is all white, treat as empty drawing
         return None, "Canvas is blank. Draw something!"


    # 2. Apply throttling to limit API calls
    if not throttler.throttle():
        # If throttled, don't call the API. Return None for the image output
        # and a status message. The output image will remain unchanged from
        # the last successful generation in the UI.
        return None, f"Throttled: Please wait {THROTTLE_TIME}s between updates."

    print("Processing drawing for generation...")
    # 3. Trigger asynchronous image generation function
    # Await the async generation call. It returns (PIL_Image or None, status_string)
    generated_image, status_message = await generate_image_from_drawing(pil_image, current_prompt)

    # 4. Return the results to Gradio outputs
    # If generated_image is None (due to error or throttling), the output gr.Image will handle it
    return generated_image, status_message

def clear_canvas():
    """Clears the drawing canvas and status."""
    # Returning None for gr.Image clears it. Clear status text too.
    return None, "Canvas cleared. Ready for a new sketch!"

def update_analysis_display_sync(canvas_input_data):
    """
    Synchronous function to update the detected elements display quickly.
    Called with queue=False.
    """
    if canvas_input_data is None or not canvas_input_data.get('image', None) is not None:
        return "Draw something to see analysis."

    pil_img = numpy_to_pil(canvas_input_data)
    if pil_img is None:
        return "Error analyzing drawing."

    # Check if the image is mostly white (empty drawing)
    if np.all(pil_to_numpy(pil_img) == 255):
         return "Canvas is blank. Draw something!"


    # Perform analysis
    shapes, colors = analyze_drawing(pil_img)

    # Format the output string
    shape_text = f"Shapes: {', '.join(shapes) or 'None detected'}"
    color_text = f"Colors: {', '.join(colors) or 'None detected'}"
    return f"{shape_text}\n{color_text}"


# --- Gradio Blocks Definition ---
print("Setting up Gradio Blocks...")

# Using Blocks for flexible layout
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🎨 Real-time Sketch to Image AI Generator
        Draw on the canvas and watch AI generate an image based on your sketch!
        <br>
        *(Generation is throttled to avoid overwhelming the API. Drawing analysis updates faster.)*
        <br>
        *Powered by Hugging Face Inference API*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Input drawing canvas component
            canvas = gr.Image(
                label="Draw Here",
                type="numpy", # Input will be a numpy array (or dict in 4.x)
                image_mode="RGB", # Ensure color format is RGB
                height=512,
                width=512,
                interactive=True, # Allow drawing
                # brush_radius and tool_kwargs are NOT set here in recent Gradio versions
            )
            with gr.Row():
                # Clear button
                clear_button = gr.Button("Clear Canvas")
                # Note: Complex undo requires storing drawing history, which is more advanced.

            # Optional text prompt input
            prompt_input = gr.Textbox(
                label="Optional Text Prompt",
                placeholder="e.g., 'a red apple on a table', 'a robot cat in space'",
                lines=2 # Allow multiple lines of text
            )

            # Display area for detected sketch elements
            sketch_analysis_display = gr.Textbox(
                label="Detected Sketch Elements",
                interactive=False, # Display only, not editable by user
                lines=3, # Show multiple lines
                max_lines=5 # Prevent excessive growth
            )

        with gr.Column(scale=1):
            # Output generated image component
            output_image = gr.Image(
                label="Generated Image",
                height=512,
                width=512,
                interactive=False # Display only
            )
            # Status message display area
            status_display = gr.Textbox(
                label="Status",
                interactive=False, # Di
                lines=2, # Show multiple lines
                max_lines=4 # Prevent excessive growth
            )

    # --- Event Listeners ---

    # 1. Trigger image generation when the canvas changes OR the text prompt is submitted
    # Use the .change() event for the canvas
    canvas.change(
        fn=process_drawing_and_generate, # The async function to call
        inputs=[canvas, prompt_input],
        outputs=[output_image, status_display],
        # queue=True is essential for async functions and handling rapid UI updates
        queue=True
    )
    # Also trigger generation when the prompt textbox has a value submitted (e.g., user presses Enter)
    prompt_input.submit(
        fn=process_drawing_and_generate,
        inputs=[canvas, prompt_input],
        outputs=[output_image, status_display],
        queue=True # Queue this event as well
    )

    # 2. Trigger a separate, faster update for the analysis display
    # Use the .change() event for the canvas again, but with a different function
    canvas.change(
        fn=update_analysis_display_sync, # The synchronous analysis function
        inputs=[canvas],
        outputs=[sketch_analysis_display],
        queue=False # Process this event immediately without waiting for the main queue
    )

    # 3. Clear button event listener
    clear_button.click(
        fn=clear_canvas,
        inputs=[], # No inputs needed to clear
        outputs=[canvas, status_display, output_image, sketch_analysis_display], # Clear multiple components
        queue=False # Clearing is fast, process immediately
    )


    gr.Markdown(f"--- \n*Model: `{HF_MODEL}` | Generation Throttle: `{THROTTLE_TIME}s`*")

print("Gradio Blocks defined.")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # .queue() must be called to enable queueing on events
    demo.queue()
    # .launch() starts the web server. share=True creates a public link required for Spaces.
    # debug=True can be helpful during local development for detailed errors.
    demo.launch(share=True)
    print("Gradio App Launched.")