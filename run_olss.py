from diffusers import DiffusionPipeline
import torch
from FastSDXL.BlockUNet import BlockUNet
from FastSDXL.OLSS import SchedulerWrapper, OLSSScheduler
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
block_unet = BlockUNet().half()
block_unet.from_diffusers(state_dict=pipe.unet.state_dict())
pipe.unet = block_unet
pipe.enable_model_cpu_offload()


# Train
train_steps = 300
inference_steps = 30
pipe.scheduler = SchedulerWrapper(DDIMScheduler.from_config(pipe.scheduler.config))
pipe(
    prompt="cinematic still a dog. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    height=1024, width=1024, num_inference_steps=train_steps
)
pipe(
    prompt="cinematic still a cat. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    height=1024, width=1024, num_inference_steps=train_steps
)
pipe(
    prompt="cinematic still a woman. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    height=1024, width=1024, num_inference_steps=train_steps
)
pipe(
    prompt="cinematic still a car. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    height=1024, width=1024, num_inference_steps=train_steps
)
pipe.scheduler.prepare_olss(inference_steps)
pipe.scheduler.olss_scheduler.save("models/olss_scheduler.bin")


# Test
prompt = "cinematic still a forest in spring, birds. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

torch.manual_seed(0)
pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0/scheduler")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=1024, width=1024, num_inference_steps=inference_steps).images[0]
image.save(f"dpmsolver.png")

torch.manual_seed(0)
pipe.scheduler = OLSSScheduler.load("models/olss_scheduler.bin")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=1024, width=1024, num_inference_steps=inference_steps).images[0]
image.save(f"olss.png")

torch.manual_seed(0)
pipe.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0/scheduler")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=1024, width=1024, num_inference_steps=inference_steps).images[0]
image.save(f"ddim.png")
