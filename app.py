# app.py

"""
Sketch-to-Image Generator
This application provides the user interface for generating images from sketches
using the Gradio library. The backend logic is handled by image_generator.py.
"""

import gradio as gr
import logging
from PIL import Image

import config
from image_generator import generate_image, numpy_to_pil, inpaint_image

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gradio Interface Definition ---
def create_gradio_interface():
    """Defines and returns the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Sketch-to-Image Generator")
        gr.Markdown("Create new images from sketches, uploads, and text prompts.")
        gr.Markdown(
            "Use the sketch canvas tools to draw with a brush, erase, and undo/redo. "
            "Use the controls below to customize brush size, brush color, and canvas size."
        )

        with gr.Tabs():
            with gr.TabItem("Create"):
                with gr.Row():
                    brush_size = gr.Slider(minimum=1, maximum=80, step=1, value=8, label="Brush Size")
                    brush_color = gr.ColorPicker(value="#000000", label="Brush Color")
                    canvas_size = gr.Dropdown(
                        choices=["512x512", "768x768", "1024x1024"],
                        value="512x512",
                        label="Canvas Size",
                    )
                    transparent_background = gr.Checkbox(value=False, label="Transparent Background")
                    apply_canvas_settings = gr.Button("Apply Canvas Settings")
                with gr.Row():
                    canvas = gr.ImageEditor(
                        label="Sketch Canvas",
                        type="numpy",
                        image_mode="RGBA",
                        height=512,
                        width=512,
                        brush=gr.Brush(default_size=8, colors=["#000000"], color_mode="fixed"),
                        eraser=gr.Eraser(default_size=20),
                        layers=False,
                        sources=("upload", "clipboard"),
                    )
                    output_image = gr.Image(
                        label="Generated Image",
                        interactive=False,
                        height=512,
                        width=512
                    )
                prompt_input = gr.Textbox(label="Text Prompt", placeholder="Describe the image you want to generate (e.g., 'A photorealistic cat')")

                with gr.Accordion("Generation History", open=True):
                    history_gallery = gr.Gallery(label="Your recent creations", height=256)

            with gr.TabItem("Advanced Settings"):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="Describe what you DON'T want in the image (e.g., 'blurry, low quality')")
                guidance_scale_slider = gr.Slider(minimum=0, maximum=20, step=0.5, value=7.5, label="Guidance Scale (CFG)")
                seed_input = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                gr.Markdown("Set Seed to -1 for a random generation.")

            with gr.TabItem("Inpainting"):
                gr.Markdown("Upload an image, mask the area you want to change, and provide a new prompt.")
                with gr.Row():
                    inpainting_image = gr.Image(
                        label="Image to Edit",
                        type="pil",
                        height=512,
                        width=512
                    )
                    inpainting_output = gr.Image(
                        label="Inpainted Result",
                        interactive=False,
                        height=512,
                        width=512
                    )
                inpainting_prompt = gr.Textbox(label="Inpainting Prompt", placeholder="Describe what you want to see in the masked area")
                inpainting_button = gr.Button("Regenerate Masked Area", variant="primary")

        status_display = gr.Textbox(label="Status", interactive=False, lines=1)
        generate_button = gr.Button("Generate Image", variant="primary")

        # --- State Management ---
        history_state = gr.State([])

        def update_canvas_settings(size_preset, size, color, is_transparent):
            width, height = map(int, size_preset.split("x"))
            background = None if is_transparent else Image.new("RGBA", (width, height), (255, 255, 255, 255))
            return gr.ImageEditor(
                value=background,
                image_mode="RGBA",
                height=height,
                width=width,
                brush=gr.Brush(default_size=int(size), colors=[color], color_mode="fixed"),
                eraser=gr.Eraser(default_size=max(8, int(size * 1.5))),
                layers=False,
                sources=("upload", "clipboard"),
            )

        # --- Event Handling ---
        async def on_generate(image_dict, prompt, negative_prompt, guidance_scale, seed, current_history):
            """Wrapper to handle the image generation process and update history."""
            if image_dict is None:
                return None, "Please draw something on the canvas first.", current_history

            pil_image = numpy_to_pil(image_dict)
            generated_img, status = await generate_image(
                pil_image=pil_image,
                prompt_override=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                seed=int(seed)
            )

            if generated_img:
                current_history.insert(0, generated_img)
                if len(current_history) > 10:
                    current_history.pop()

            return generated_img, status, current_history

        generate_button.click(
            fn=on_generate,
            inputs=[canvas, prompt_input, negative_prompt_input, guidance_scale_slider, seed_input, history_state],
            outputs=[output_image, status_display, history_gallery]
        )
        apply_canvas_settings.click(
            fn=update_canvas_settings,
            inputs=[canvas_size, brush_size, brush_color, transparent_background],
            outputs=canvas,
        )

        inpainting_button.click(
            fn=inpaint_image,
            inputs=[inpainting_image, inpainting_prompt],
            outputs=[inpainting_output, status_display]
        )

    return demo

# --- Launch Application ---
if __name__ == "__main__":
    app_interface = create_gradio_interface()
    app_interface.launch(share=True)
