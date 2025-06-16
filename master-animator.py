import argparse
import os
import sys # Import sys to get the current Python executable
import uuid
import subprocess
from PIL import Image
import shutil # For removing the directory in case of error during setup

from pipeline import initialize_diffusion_pipeline # Import the new function
import torch # For torch.cuda.is_available()

def main():
    parser = argparse.ArgumentParser(description="Master animator for diffusion-based animations.")
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID from Hugging Face or path to a local model directory."
    )
    parser.add_argument(
        "--lora",
        type=str,
        action='append', # Allows specifying multiple LoRAs
        help="Path to LoRA file (.safetensors). Can be used multiple times for multiple LoRAs. "
             "Example: --lora ./pixel_sprites.safetensors"
    )
    # Removed --lora_weight_name argument
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to guide the initial image generation and animation."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Total number of frames for the animation (e.g., 0.png to 7.png for 8 frames). Default is 8."
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="Strength for the img2img process (0.0 to 1.0). Default is 0.5."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for diffusion. Default is 7.5."
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=30, # Reduced default for faster iteration
        help="Number of inference steps for diffusion. Default is 30."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the initial generated image. Default is 512."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the initial generated image. Default is 512."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output animation player. Default is 10."
    )
    # SDXL specific arguments (no longer Lightning specific)
    parser.add_argument(
        "--sdxl_variant",
        type=str,
        default=None,
        help="Variant for SDXL pipeline (e.g., 'fp16'). Applicable if an SDXL model is used."
    )
    parser.add_argument(
        "--scheduler_timestep_spacing",
        type=str,
        default=None,
        help="Scheduler timestep spacing (e.g., 'trailing', 'leading', 'linspace')."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simplified LoRA handling: args.lora is now expected to be a list of file paths
    lora_file_paths = args.lora if args.lora else []

    # print(f"Attempting to initialize pipeline with model: {args.model}")
    # if lora_file_paths:
    #     print("LoRA files to load:")
    #     for lora_file in lora_file_paths:
    #         print(f"  - {lora_file}")
    
    # --- 1. Create Output Directory ---
    random_suffix = uuid.uuid4().hex[:8]
    output_folder_name = f"animate_{random_suffix}"
    try:
        os.makedirs(output_folder_name, exist_ok=True)
        print(f"Animation frames will be saved to: {output_folder_name}")
    except Exception as e:
        print(f"Error creating output directory {output_folder_name}: {e}")
        return

    base_image_path = os.path.join(output_folder_name, "base.png")
    first_animation_frame_path = os.path.join(output_folder_name, "0.png")

    # --- 2. Initial Image Generation (txt2img) ---
    print("\n--- Initializing Text-to-Image Pipeline ---")
    txt2img_pipe = initialize_diffusion_pipeline(
        model_id_or_path=args.model,
        lora_file_paths=lora_file_paths,
        device_str=device,
        pipeline_type="txt2img",
        sdxl_variant=args.sdxl_variant,
        scheduler_timestep_spacing=args.scheduler_timestep_spacing
    )

    if not txt2img_pipe:
        print("Failed to initialize Text-to-Image pipeline. Exiting.")
        shutil.rmtree(output_folder_name) # Clean up created directory
        return

    print(f"\n--- Generating Base Image ({args.width}x{args.height}) ---")
    print(f"Prompt: {args.prompt}")
    txt2img_pipe.enable_sequential_cpu_offload()
    try:
        with torch.inference_mode():
            image = txt2img_pipe(
                prompt=args.prompt,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height
            ).images[0]
        image.save(base_image_path)
        print(f"Base image saved as {base_image_path}")
    except Exception as e:
        print(f"Error during base image generation: {e}")
        shutil.rmtree(output_folder_name)
        return
    # Unload txt2img_pipe if not needed further to free memory, or rely on Python's GC
    del txt2img_pipe
    if device == "cuda":
        torch.cuda.empty_cache()


    # --- 3. Background Removal ---
    print(f"\n--- Removing Background from Base Image ---")
    try:
        python_executable = sys.executable # Get the path to the current Python interpreter
        rembg_cmd = [
            python_executable, "back-remove.py",
            base_image_path,
            "--output_image", first_animation_frame_path
        ]
        print(f"Executing: {' '.join(rembg_cmd)}")
        result = subprocess.run(rembg_cmd, check=True, capture_output=True, text=True)
        print("Background removal script output:")
        print(result.stdout)
        if not os.path.exists(first_animation_frame_path):
            print(f"Error: Background removal did not produce {first_animation_frame_path}")
            if result.stderr: print(f"Error from script: {result.stderr}")
            shutil.rmtree(output_folder_name)
            return
        print(f"Background removed, first animation frame saved as {first_animation_frame_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during background removal: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        shutil.rmtree(output_folder_name)
        return
    except FileNotFoundError:
        print("Error: back-remove.py not found. Make sure it's in the same directory or in PATH.")
        shutil.rmtree(output_folder_name)
        return


    # --- 4. Sequential Animation Frames (Img2Img) ---
    if args.num_frames > 0: # Only proceed if frames are requested (0.png is one frame)
        print("\n--- Initializing Image-to-Image Pipeline for Animation ---")
        img2img_pipe = initialize_diffusion_pipeline(
            model_id_or_path=args.model,
            lora_file_paths=lora_file_paths,
            device_str=device,
            pipeline_type="img2img",
            sdxl_variant=args.sdxl_variant,
            scheduler_timestep_spacing=args.scheduler_timestep_spacing
        )
        img2img_pipe.enable_sequential_cpu_offload()
        if not img2img_pipe:
            print("Failed to initialize Image-to-Image pipeline. Exiting.")
            shutil.rmtree(output_folder_name)
            return

        print(f"\n--- Generating Animation Frames (Total: {args.num_frames}) ---")
        try:
            current_pil_image = Image.open(first_animation_frame_path).convert("RGB")
            
            # We already have 0.png, so generate num_frames - 1 more images
            for i in range(1, args.num_frames):
                frame_filename = f"{i}.png"
                frame_output_path = os.path.join(output_folder_name, frame_filename)
                print(f"Generating frame {i} ({frame_filename})...")
                
                with torch.inference_mode():
                    generated_images = img2img_pipe(
                        prompt=args.prompt, # Use the same main prompt
                        image=current_pil_image,
                        strength=args.strength,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.inference_steps
                    ).images
                
                if generated_images:
                    current_pil_image = generated_images[0]
                    current_pil_image.save(frame_output_path)
                    print(f"Saved frame {i} as {frame_output_path}")
                else:
                    print(f"Failed to generate frame {i}. Stopping.")
                    break
        except Exception as e:
            print(f"Error during animation frame generation: {e}")
            shutil.rmtree(output_folder_name)
            return
        finally:
            del img2img_pipe # Ensure pipeline is deleted
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        print("num_frames is 0, only the initial background-removed image (0.png) was created.")


    # --- 5. Cleanup Base Image ---
    if os.path.exists(base_image_path):
        try:
            os.remove(base_image_path)
            print(f"\nCleaned up base image: {base_image_path}")
        except Exception as e:
            print(f"Warning: Could not delete base image {base_image_path}: {e}")


    # --- 6. Play Animation ---
    print(f"\n--- Playing Animation ---")
    try:
        # python_executable is defined above when setting up rembg_cmd
        player_cmd = [
            python_executable, "animation-player.py",
            output_folder_name,
            "--fps", str(args.fps)
        ]
        print(f"Executing: {' '.join(player_cmd)}")
        subprocess.run(player_cmd) # Not checking for errors, player might be closed by user
        print("Animation player finished or was closed.")
    except FileNotFoundError:
        print("Error: animation-player.py not found. Make sure it's in the same directory or in PATH.")
    except Exception as e:
        print(f"Error trying to run animation player: {e}")
        
    print("\nMaster Animator process complete.")

if __name__ == "__main__":
    main()