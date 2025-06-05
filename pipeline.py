import torch
import os
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def initialize_diffusion_pipeline(
    model_id_or_path,
    lora_file_paths=None,
    device_str="cuda",
    pipeline_type="img2img",
    sdxl_lightning_unet_repo_id=None,
    sdxl_lightning_unet_ckpt=None,
    sdxl_variant=None, # e.g., "fp16"
    scheduler_timestep_spacing=None # e.g., "trailing"
):
    """
    Initializes and configures a Stable Diffusion or SDXL pipeline.

    Args:
        model_id_or_path (str): Hugging Face model ID or local path to the base model.
        lora_file_paths (list of str, optional): List of file paths to LoRA (.safetensors) files.
        device_str (str): "cuda" or "cpu".
        pipeline_type (str): "txt2img" or "img2img".
        sdxl_lightning_unet_repo_id (str, optional): Repo ID for SDXL-Lightning UNet.
        sdxl_lightning_unet_ckpt (str, optional): Checkpoint filename for SDXL-Lightning UNet.
        sdxl_variant (str, optional): Variant for SDXL pipeline (e.g., "fp16").
        scheduler_timestep_spacing (str, optional): Timestep spacing for EulerDiscreteScheduler.

    Returns:
        A configured StableDiffusionPipeline or StableDiffusionImg2ImgPipeline object, or None if initialization fails.
    """
    torch_dtype = torch.float16 if device_str == "cuda" else torch.float32
    pipe = None
    is_sdxl_lightning = sdxl_lightning_unet_repo_id and sdxl_lightning_unet_ckpt

    if is_sdxl_lightning:
        print(f"Attempting to load SDXL-Lightning model: {model_id_or_path} with UNet from {sdxl_lightning_unet_repo_id}/{sdxl_lightning_unet_ckpt}")
        sdxl_pipeline_class = StableDiffusionXLImg2ImgPipeline if pipeline_type == "img2img" else StableDiffusionXLPipeline
        
        try:
            print(f"Loading UNet config from base: {model_id_or_path}")
            unet = UNet2DConditionModel.from_config(model_id_or_path, subfolder="unet").to(device_str, torch_dtype)
            print(f"Downloading and loading UNet checkpoint: {sdxl_lightning_unet_repo_id}/{sdxl_lightning_unet_ckpt}")
            unet_checkpoint_path = hf_hub_download(sdxl_lightning_unet_repo_id, sdxl_lightning_unet_ckpt)
            unet.load_state_dict(load_file(unet_checkpoint_path, device=device_str))
            
            print(f"Loading SDXL pipeline: {model_id_or_path} with custom UNet.")
            pipe = sdxl_pipeline_class.from_pretrained(
                model_id_or_path,
                unet=unet,
                torch_dtype=torch_dtype,
                variant=sdxl_variant or ("fp16" if device_str == "cuda" else None), # Default variant for SDXL
                safety_checker=None
            )
            print("Successfully loaded SDXL-Lightning pipeline.")
        except Exception as e:
            print(f"Error loading SDXL-Lightning model: {e}")
            return None
        # Default scheduler spacing for SDXL Lightning
        if scheduler_timestep_spacing is None:
            scheduler_timestep_spacing = "trailing"

    else:
        print(f"Attempting to load standard Stable Diffusion model: {model_id_or_path} as {pipeline_type} pipeline.")
        sd_pipeline_class = StableDiffusionImg2ImgPipeline if pipeline_type == "img2img" else StableDiffusionPipeline
        try:
            pipe = sd_pipeline_class.from_pretrained(
                model_id_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
            print(f"Successfully loaded model '{model_id_or_path}' using from_pretrained.")
        except (OSError, IOError) as e:
            print(f"Could not load model '{model_id_or_path}' directly with from_pretrained: {e}")
            if os.path.isdir(model_id_or_path):
                print(f"'{model_id_or_path}' is a directory. Attempting to load from local directory.")
                try:
                    pipe = sd_pipeline_class.from_pretrained(
                        model_id_or_path,
                        local_files_only=True,
                        torch_dtype=torch_dtype,
                        safety_checker=None
                    )
                    print(f"Successfully loaded model from local directory: {model_id_or_path}")
                except Exception as e_local:
                    print(f"Failed to load model from local directory '{model_id_or_path}': {e_local}")
                    return None
            else:
                print(f"'{model_id_or_path}' is not a recognized Hugging Face ID or local directory. Model loading failed.")
                return None
        except Exception as e_general:
            print(f"An unexpected error occurred while loading model '{model_id_or_path}': {e_general}")
            return None

    if not pipe:
        print("Pipeline could not be initialized.")
        return None

    pipe.to(device_str)

    if scheduler_timestep_spacing:
        print(f"Setting scheduler timestep_spacing to: {scheduler_timestep_spacing}")
        try:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing=scheduler_timestep_spacing)
        except Exception as e:
            print(f"Warning: Could not set custom scheduler timestep_spacing: {e}. Using default scheduler configuration.")

    if lora_file_paths:
        print(f"Loading {len(lora_file_paths)} LoRA(s)...")
        for lora_path in lora_file_paths:
            if not lora_path or not os.path.exists(lora_path):
                print(f"LoRA file not found or path is invalid: '{lora_path}'. Skipping.")
                continue
            if not lora_path.endswith(".safetensors"):
                print(f"LoRA file '{lora_path}' does not end with .safetensors. Skipping.")
                continue

            try:
                lora_file_dir = os.path.dirname(lora_path)
                lora_filename = os.path.basename(lora_path)
                if not lora_file_dir: # If it's just a filename (e.g., "lora.safetensors")
                    lora_file_dir = "." # Assume current directory

                # adapter_name can be specified if you want to load multiple LoRAs and switch between them
                # For simplicity here, we'll let diffusers assign a default or use the filename.
                # Or, you can generate one: adapter_name = lora_filename.replace(".safetensors", "")
                print(f"Loading LoRA '{lora_filename}' from directory '{lora_file_dir}'...")
                pipe.load_lora_weights(lora_file_dir, weight_name=lora_filename) # Removed adapter_name for simplicity
                print(f"Successfully loaded LoRA: {lora_path}")
            except Exception as e:
                print(f"Could not load LoRA '{lora_path}': {e}. Proceeding without this LoRA.")
    else:
        print("No LoRAs specified or to load.")

    print("Pipeline initialization complete.")
    return pipe

if __name__ == '__main__':
    # Example Usage (for testing pipeline.py directly)
    print("Testing pipeline.py...")

    # Test 1: Default SD 1.5 from Hugging Face, no LoRA
    print("\n--- Test 1: Default SD 1.5 (txt2img) ---")
    pipe1 = initialize_diffusion_pipeline("runwayml/stable-diffusion-v1-5", pipeline_type="txt2img")
    if pipe1:
        print("Test 1 Succeeded: Pipeline object created.")
    else:
        print("Test 1 Failed.")

    # Test 2: Default SD 1.5 (img2img) with a hypothetical local LoRA
    # Ensure 'pixel sprites.safetensors' exists in the current directory for this to work
    print("\n--- Test 2: Default SD 1.5 (img2img) with pixel sprites LoRA ---")
    # Ensure 'pixel sprites.safetensors' exists in the current directory or provide full path
    lora_files_to_load = ["./pixel sprites.safetensors"]
    # Example with full path:
    # lora_files_to_load = ["C:/path/to/your/loras/pixel_sprites.safetensors"]
    
    if not os.path.exists(lora_files_to_load[0]):
        print(f"Test LoRA file {lora_files_to_load[0]} not found. Skipping Test 2 LoRA loading.")
        lora_files_to_load = None # Set to None if file doesn't exist to avoid error in test

    pipe2 = initialize_diffusion_pipeline(
        "runwayml/stable-diffusion-v1-5",
        lora_file_paths=lora_files_to_load,
        pipeline_type="img2img"
    )
    if pipe2:
        print("Test 2 Succeeded: Pipeline object created.")
        # You might want to unload LoRA weights if you plan to reuse pipe2 for other tests without this LoRA
        # pipe2.unload_lora_weights()
    else:
        print("Test 2 Failed.")

    # Test 3: Non-existent model
    print("\n--- Test 3: Non-existent model ---")
    pipe3 = initialize_diffusion_pipeline("this/model-does-not-exist")
    if not pipe3:
        print("Test 3 Succeeded: Pipeline creation correctly failed.")
    else:
        print("Test 3 Failed: Pipeline object was created for a non-existent model.")

    # Test 4: Loading a model that might be local (if you have one, e.g., "C:/models/my-sd-model")
    # print("\n--- Test 4: Local model path (replace with actual path) ---")
    # pipe4 = initialize_diffusion_pipeline("path/to/your/local/sd/model/directory") # Standard SD
    # if pipe4:
    #     print("Test 4 Succeeded.")
    # else:
    #     print("Test 4 Failed.")

    # Test 5: SDXL Lightning 4-step
    print("\n--- Test 5: SDXL Lightning 4-step (txt2img) ---")
    # Base SDXL model ID
    sdxl_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # SDXL Lightning UNet details
    unet_repo = "ByteDance/SDXL-Lightning"
    unet_ckpt = "sdxl_lightning_4step_unet.safetensors"

    pipe5 = initialize_diffusion_pipeline(
        model_id_or_path=sdxl_base_model_id,
        pipeline_type="txt2img",
        sdxl_lightning_unet_repo_id=unet_repo,
        sdxl_lightning_unet_ckpt=unet_ckpt,
        sdxl_variant="fp16", # Common for SDXL
        # scheduler_timestep_spacing="trailing" # Will default to trailing for SDXL-Lightning
    )
    if pipe5:
        print("Test 5 Succeeded: SDXL Lightning Pipeline object created.")
        # Example generation (optional, can be slow)
        # prompt_test_sdxl = "A majestic lion in a futuristic city, detailed, cinematic"
        # with torch.inference_mode():
        #     image_sdxl = pipe5(prompt=prompt_test_sdxl, num_inference_steps=4, guidance_scale=0).images[0] # guidance_scale=0 for lightning
        # image_sdxl.save("test_sdxl_lightning.png")
        # print("Test SDXL image saved as test_sdxl_lightning.png")
    else:
        print("Test 5 Failed.")

    print("\nPipeline.py testing finished.")