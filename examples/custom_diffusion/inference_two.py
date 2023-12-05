import torch
from diffusers import StableDiffusionPipeline
import os

save_path_name = "gupta_nima_experiment_2"
token_name = "<nimsi>_<guptamvjm>"
class_name = "person"


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
# pipe.unet.load_attn_procs(
#     save_path_name, weight_name="pytorch_custom_diffusion_weights.bin"
# )
# pipe.load_textual_inversion(save_path_name, weight_name=f"<nimsi>.bin")
# pipe.load_textual_inversion(save_path_name, weight_name=f"<guptamvjm>.bin")
for j in range(10):
    image = pipe(
        f"Photograph of a dog following a cat, and the cat is walking to the right of the photo.",
        # f"Photo of two astronauts standing next to each other. There's a really tall astronaut on the left of the photo and a really short astronaut on the right of the photo.",
        # "a photograph of an astronaut riding a horse",
        # f"Person wearing blue tank top standing to the right of person wearing a pink sweater",
        # f"<nimsi> person and <guptamvjm> person standing together with a bookshelf behind them",
        # f"<nimsi> person in a watercolor painting",
        num_inference_steps=50,
        eta=1.0,
    ).images[0]
    if not os.path.exists(token_name):
        os.mkdir(token_name)
    image.save(f"{token_name}/{token_name}{j}.png")
