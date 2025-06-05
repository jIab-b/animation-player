import argparse
import torch
import os
import uuid
from PIL import Image
import shutil

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from safetensors.torch import load_file # Assuming this is still needed for LoRA if it were compatible

def setup_pipeline(device, torch_dtype, model_id="runwayml/stable-diffusion-v1-5"):
    """Sets up the Stable Diffusion pipeline."""
    try:
        # Prioritize Img2Img pipeline for the core animation loop
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
    except Exception as e:
        print(f"Could not load Img2Img pipeline: {e}. Falling back to base pipeline for initial setup check.")
        # Fallback to load base pipeline to check model availability, though Img2Img is needed
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        # If we fell back, we still need the Img2Img for actual work.
        # This is more of a check. The script will likely fail later if Img2Img can't load.
        # A more robust solution would ensure Img2Img is available or exit.
        print("Warning: Img2Img pipeline is essential. The script might not function as intended.")

    pipeline.to(device)
    # pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config) # Using default
    return pipeline

def animate_frames(input_image_path, num_frames, prompt, strength, lora_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        return

    # --- Setup output directory ---
    random_suffix = uuid.uuid4().hex[:8]
    output_folder_name = f"animate_{random_suffix}"
    os.makedirs(output_folder_name, exist_ok=True)
    print(f"Saving animation frames to: {output_folder_name}")

    # --- Copy initial image (frame 0) ---
    try:
        initial_image_pil = Image.open(input_image_path).convert("RGB")
        initial_frame_output_path = os.path.join(output_folder_name, "0.png")
        initial_image_pil.save(initial_frame_output_path)
        print(f"Saved initial frame as {initial_frame_output_path}")
    except Exception as e:
        print(f"Error processing initial image: {e}")
        return

    # --- Setup pipeline ---
    pipeline = setup_pipeline(device, torch_dtype)
    if not isinstance(pipeline, StableDiffusionImg2ImgPipeline):
        # Ensure we actually have an Img2Img pipeline to work with
        print("Error: Failed to load StableDiffusionImg2ImgPipeline. Cannot proceed with animation.")
        # Attempt to load it explicitly if the initial setup didn't prioritize it correctly
        try:
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", # or your model_id
                torch_dtype=torch_dtype,
                safety_checker=None
            ).to(device)
            print("Successfully loaded Img2Img pipeline on second attempt.")
        except Exception as e_img2img:
            print(f"Critical Error: Could not load Img2Img pipeline: {e_img2img}. Exiting.")
            return


    # --- Load LoRA (if specified and compatible) ---
    if lora_path and os.path.exists(lora_path):
        try:
            # Note: LoRA loading might require specific handling depending on the LoRA's format
            # and compatibility with the base model and diffusers version.
            # The method pipeline.load_lora_weights() is standard.
            lora_directory = os.path.dirname(lora_path)
            if not lora_directory: # If lora_path is just a filename, dirname is empty
                lora_directory = "."
            pipeline.load_lora_weights(lora_directory, weight_name=os.path.basename(lora_path))
            print(f"Successfully loaded LoRA: {lora_path} from directory: {lora_directory}")
        except Exception as e: # Added except block
            print(f"Could not load LoRA '{lora_path}': {e}. Proceeding without LoRA.")
    elif lora_path:
        print(f"LoRA path specified but not found: {lora_path}. Proceeding without LoRA.")

    # --- Generate subsequent frames ---
    current_image_pil = initial_image_pil

    for i in range(1, num_frames):
        print(f"Generating frame {i}...")
        with torch.inference_mode():
            generated_images = pipeline(
                prompt=prompt,
                image=current_image_pil,
                strength=strength,
                guidance_scale=7.5, # Standard for SD 1.5
                num_inference_steps=50 # Typical for SD 1.5
            ).images
        
        if generated_images:
            current_image_pil = generated_images[0]
            frame_output_path = os.path.join(output_folder_name, f"{i}.png")
            current_image_pil.save(frame_output_path)
            print(f"Saved frame {i} as {frame_output_path}")
        else:
            print(f"Failed to generate frame {i}. Stopping.")
            break
    
    print("Animation generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animation frames from an initial image using Stable Diffusion.")
    parser.add_argument("input_image", type=str, help="Path to the initial input image.")
    parser.add_argument("prompt", type=str, help="Prompt to guide the animation.")
    parser.add_argument("--num_frames", type=int, default=8, help="Total number of frames to generate (including the initial one, so N-1 new frames). Default is 8.")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength for the img2img process (0.0 to 1.0). Default is 0.5.")
    parser.add_argument("--lora_path", type=str, default="pixel sprites.safetensors", help="Path to the LoRA file (e.g., 'pixel sprites.safetensors'). Will try to load if provided and exists.")
    
    args = parser.parse_args()

    if args.num_frames <= 1 and args.input_image:
        print("Number of frames must be greater than 1 to generate new frames. Only copying initial image.")
        # Simplified logic for num_frames=1: just copy initial image to a new folder
        random_suffix = uuid.uuid4().hex[:8]
        output_folder_name = f"animate_{random_suffix}"
        os.makedirs(output_folder_name, exist_ok=True)
        initial_frame_output_path = os.path.join(output_folder_name, "0.png")
        try:
            shutil.copy(args.input_image, initial_frame_output_path)
            print(f"Copied initial frame to {initial_frame_output_path} in {output_folder_name}")
        except Exception as e:
            print(f"Error copying initial image: {e}")
    elif args.num_frames > 0 : # Ensure num_frames is positive
         animate_frames(args.input_image, args.num_frames, args.prompt, args.strength, args.lora_path)
    else:
        print("Number of frames must be a positive integer.")