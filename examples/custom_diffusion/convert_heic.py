import os
from PIL import Image
import pyheif


def convert_heic_to_jpeg(heic_folder, output_folder, crop=True):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all HEIC files in the input folder
    heic_files = [f for f in os.listdir(heic_folder) if f.lower().endswith(".heic")]
    i = 0
    for heic_file in heic_files:
        i += 1
        print(f"Converting file: {i} of {len(heic_files)}")
        heic_path = os.path.join(heic_folder, heic_file)
        output_path = os.path.join(output_folder, os.path.splitext(heic_file)[0] + ".jpg")

        # Open the HEIC file
        heic_image = pyheif.read(heic_path)

        # Convert to Pillow Image
        pil_image = Image.frombytes(
            heic_image.mode, 
            heic_image.size, 
            heic_image.data,
            "raw",
            heic_image.mode,
            heic_image.stride,
        )

        if crop:
            # Crop the image to be square
            size = min(pil_image.size)
            left = (pil_image.width - size) / 2
            top = (pil_image.height - size) / 2
            right = (pil_image.width + size) / 2
            bottom = (pil_image.height + size) / 2
            pil_image = pil_image.crop((left, top, right, bottom))

        # Save the image as JPEG
        pil_image.convert("RGB").save(output_path, "JPEG")

if __name__ == "__main__":
    # Specify the input folder containing HEIC files
    input_folder = "/home/pingpong-michael/182/182-diffusers/examples/custom_diffusion/theBoisDataset"

    # Specify the output folder for JPEG files
    output_folder = "/home/pingpong-michael/182/182-diffusers/examples/custom_diffusion/CROPPEDtheBoisDataset"

    convert_heic_to_jpeg(input_folder, output_folder)
