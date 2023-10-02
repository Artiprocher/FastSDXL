import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
from diffusers import DiffusionPipeline
import torch
from FastSDXL.BlockUNet import BlockUNet
from FastSDXL.Styler import styles
from FastSDXL.OLSS import OLSSScheduler


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
block_unet = BlockUNet().half().to("cuda")
block_unet.from_diffusers(state_dict=pipe.unet.state_dict())
pipe.unet = block_unet
pipe.scheduler = OLSSScheduler.load("models/olss_scheduler.bin")
pipe.enable_model_cpu_offload()


def generate_image(prompt, negative_prompt, height, width, style_name="Default (Slightly Cinematic)", denoising_steps=30):
    height = (height + 63) // 64 * 64
    width = (width + 63) // 64 * 64
    for style in styles:
        if style["name"] == style_name:
            prompt = style["prompt"].replace("{prompt}", prompt)
            negative_prompt = style["negative_prompt"] + negative_prompt
            break
    print("Prompt:", prompt)
    print("Negative prompt:", negative_prompt)
    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=denoising_steps).images[0]
    return image

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative prompt")
            height = gr.Slider(label="Height", minimum=512, maximum=2048, value=1024, step=64)
            width = gr.Slider(label="Width", minimum=512, maximum=2048, value=1024, step=64)
            button = gr.Button(label="Generate")
        with gr.Column():
            image = gr.Image(label="Generated image")
    button.click(fn=generate_image, inputs=[prompt, negative_prompt, height, width], outputs=[image])

demo.queue()
demo.launch()
