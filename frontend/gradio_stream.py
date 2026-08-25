"""
frontend/gradio_stream.py
A lightweight Gradio demo intended for Colab (T4) testing.
This uses Hugging Face Diffusers when available (user must provide HF_TOKEN)
"""

import os
from PIL import Image
import numpy as np
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Try to import diffusers and accelerate; if not available, the notebook will install them.
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False


def load_pipeline(model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("diffusers not available; install diffusers, accelerate, transformers")
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


# Simple helper to convert numpy canvas to PIL

def numpy_to_pil(numpy_image_dict):
    if numpy_image_dict is None:
        return None
    numpy_array = numpy_image_dict.get('image') if isinstance(numpy_image_dict, dict) else numpy_image_dict
    if isinstance(numpy_array, np.ndarray):
        if numpy_array.ndim == 3:
            return Image.fromarray(numpy_array[:, :, :3].astype('uint8'))
        else:
            return Image.fromarray(numpy_array.astype('uint8')).convert('RGB')
    return None


# If diffusers installed, create a lazy-loaded pipeline
PIPE = None

def generate_with_pipeline(pil_image, prompt_override="A sketch-based image"):
    global PIPE
    if PIPE is None:
        PIPE = load_pipeline()
    prompt = prompt_override or "an artistic rendering of the sketch"
    image = PIPE(prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
    return image


def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Colab Sketch-to-Image Demo (T4)")
        canvas = gr.Image(label="Draw Here", type="numpy", image_mode="RGB", height=512, width=512, interactive=True)
        prompt_input = gr.Textbox(label="Text Prompt", placeholder="Optional description")
        output_image = gr.Image(label="Generated Image", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

        def _wrap(img, prompt):
            pil = numpy_to_pil(img)
            if pil is None:
                return None, "Invalid canvas"
            if DIFFUSERS_AVAILABLE and HF_TOKEN:
                return generate_with_pipeline(pil, prompt), "Generated locally via diffusers"
            else:
                return None, "Diffusers not available or HF_TOKEN missing in Colab"

        canvas.change(fn=_wrap, inputs=[canvas, prompt_input], outputs=[output_image, status])
    demo.launch(share=True)


if __name__ == '__main__':
    launch_demo()
