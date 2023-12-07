import torch
from diffusers import StableDiffusionPipeline
import os

save_path_name = "gupta_nima_crossattn"
token_name = "<nimsi>_<guptamvjm>"
class_name = "person"


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.unet.load_attn_procs(
    save_path_name, weight_name="pytorch_custom_diffusion_weights.bin"
)
pipe.load_textual_inversion(save_path_name, weight_name=f"<nimsi>.bin")
pipe.load_textual_inversion(save_path_name, weight_name=f"<guptamvjm>.bin")
for j in range(10):
    image = pipe(
        f"<nimsi> person and <guptamvjm> person walking together and holding hands. <nimsi> person is on the left of the photo and <guptamvjm> person is on the right of the photo.",
        # f"<nimsi> person in a watercolor painting",
        num_inference_steps=50,
        eta=1.0,
    ).images[0]
    if not os.path.exists(token_name):
        os.mkdir(token_name)
    # image.save(f"{token_name}/{token_name}{j}.png")
    image.save(f"{token_name}/small_set{j}.png")
