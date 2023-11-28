import torch
from diffusers import DiffusionPipeline
import os

save_path_name = "gupta_nima_experiment"
token_name = "<nimsi>_<guptamvjm>"
class_name = "person"


pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    save_path_name, weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion(save_path_name, weight_name=f"<nimsi>.bin")
pipe.load_textual_inversion(save_path_name, weight_name=f"<guptamvjm>.bin")
generator = torch.Generator(device=pipe.device).manual_seed(1000)
image = pipe(
    f"<nimsi> person teaching <guptamvjm> person about math",
    num_inference_steps=50,
    eta=1.0,
).images
if not os.path.exists(token_name):
    os.mkdir(token_name)
i = 0
for img in image:
    img.save(f"{token_name}/{token_name}{i}.png")
    i += 1