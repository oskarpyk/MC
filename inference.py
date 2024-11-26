import os.path
import pdb

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image
import argparse

from garment_adapter.garment_diffusion import ClothAdapter
from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='oms diffusion')
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--enable_cloth_guidance', action="store_true")
    parser.add_argument('--pipe_path', type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
    parser.add_argument('--output_path', type=str, default="./output_img")
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of denoising steps (higher = better quality but slower)")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale (higher = more faithful to prompt but less diverse)")

    args = parser.parse_args()

    device = "cuda"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    if args.enable_cloth_guidance:
        pipe = OmsDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.pipe_path, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    full_net = ClothAdapter(pipe, args.model_path, device, args.enable_cloth_guidance, False)
    images = full_net.generate(
        cloth_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )
    for i, image in enumerate(images[0]):
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
