# app.py

"""
Sketch-to-Image Generator
This application provides the user interface for generating images from sketches
using the Gradio library. The backend logic is handled by image_generator.py.
"""

import gradio as gr
import logging

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

        with gr.Tabs():
            with gr.TabItem("Create"):
                with gr.Row():
                    canvas = gr.Image(
                        label="Draw Here or Upload an Image",
                        type="numpy",
                        image_mode="RGB",
                        height=512,
                        width=512,
                        interactive=True,
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
