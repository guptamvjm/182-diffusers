import torch
from diffusers import DiffusionPipeline
import os

save_path_name = "path-to-save-model-dog"
token_name = "<dog1>"


pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    save_path_name, weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion(save_path_name, weight_name=f"{token_name}.bin")

image = pipe(
    f"{token_name} dog wearing sunglasses",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images
if not os.path.exists(token_name):
    os.mkdir(token_name)
i = 0
for img in image:
    img.save(f"{token_name}/dog{i}.png")
    i += 1