import argparse
import numpy as np
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline
import torch


class BlendedLatnetDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()
        torch.manual_seed(42)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--init_image", type=str, required=True, help="The path to the input image"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="timbrooks/instruct-pix2pix",
            help="The path to the HuggingFace model",
        )
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--output_path",
            type=str,
            default="outputs/res.jpg",
            help="The destination output path",
        )

        self.args = parser.parse_args()

    def load_models(self):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        ).to(self.args.device)


    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
    ):

        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        image = self.pipe(prompt=prompt, image=image).images[0]

        return image


if __name__ == "__main__":
    bld = BlendedLatnetDiffusion()
    result = bld.edit_image(
        bld.args.init_image,
        prompt=bld.args.prompt
    )
    result.save(bld.args.output_path)
