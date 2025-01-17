import torch
from diffusers import DiffusionPipeline
import os

save_path_name = "nimsi-experiment"
token_name = "<nimsi>"
class_name = "person"


pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    save_path_name, weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion(save_path_name, weight_name=f"{token_name}.bin")
if not os.path.exists(token_name):
    os.mkdir(token_name)
for j in range(3):
    image = pipe(
        f"{token_name} {class_name} in a watercolor painting",
        num_inference_steps=50,
        eta=1.0,
    ).images[0]

    image.save(f"{token_name}/{token_name}{j}.png")
