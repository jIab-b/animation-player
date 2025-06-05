import argparse
import torch
import os
import json # Added for loading prompts.jsonl
from PIL import Image
import uuid
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler # Changed XLPipeline to Pipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def gen_image(prompt_text=None):
    base = "runwayml/stable-diffusion-v1-5" # Changed to SD 1.5
    # repo = "ByteDance/SDXL-Lightning" # Removed for base SDXL
    # ckpt = "sdxl_lightning_4step_unet.safetensors" # Removed for base SDXL

    device = 'cuda'

    torch_dtype = torch.float16

    # unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch_dtype) # Not loading UNet separately
    # print(f"Downloading and loading UNet checkpoint: {repo}/{ckpt}") # Removed
    # unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device)) # Removed
    
    # Load Pipeline
    pipeline = StableDiffusionPipeline.from_pretrained( # Changed to StableDiffusionPipeline
        base,
        torch_dtype=torch_dtype,
        # variant="fp16" if device == "cuda" else None # variant not typically used for SD 1.5 like this
        safety_checker=None # Optional: Disable safety checker if desired
    )
    pipeline.to(device) # Move entire pipeline to device
    # pipeline.enable_sequential_cpu_offload() # May not be needed or could be re-enabled if memory issues
    # pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing") # Using default scheduler for SD 1.5

    # Load LoRA
    lora_path = "pixel sprites.safetensors"
    try:
        pipeline.load_lora_weights(".", weight_name=lora_path)
        print(f"Loaded LoRA: {lora_path}")
    except Exception as e:
        print(f"Could not load LoRA: {e}. Proceeding without LoRA.")


    prompt = prompt_text if prompt_text else 'a lightning ball'
    print(f"Using prompt: {prompt}") # Added for clarity
    with torch.inference_mode():
        images = pipeline(
            prompt=prompt,
            num_inference_steps=50, # Typical steps for SD 1.5
            guidance_scale=7.5, # Standard guidance scale
            width=512, # Typical SD 1.5 width
            height=512 # Typical SD 1.5 height
        ).images

    random_suffix = uuid.uuid4().hex[:8] # Generate an 8-character random hex string
    output_filename = f'img_{random_suffix}.png'
    images[0].save(output_filename)
    print(f"Image saved as {output_filename}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion with an optional prompt.")
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation.")
    args = parser.parse_args()

    gen_image(prompt_text=args.prompt)

    
    
