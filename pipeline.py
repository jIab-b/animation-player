import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler

def initialize_diffusion_pipeline(model_id_or_path, lora_details=None, device_str="cuda", pipeline_type="img2img"):
    """
    Initializes and configures a Stable Diffusion pipeline.

    Args:
        model_id_or_path (str): Hugging Face model ID or local path to the model.
        lora_details (list of dict, optional): List of LoRA details, where each dict is
                                               {'path': str, 'weight_name': str, 'adapter_name': str (optional)}.
                                               If 'weight_name' is None, 'path' is assumed to be the full path to the LoRA file.
        device_str (str): "cuda" or "cpu".
        pipeline_type (str): "txt2img" or "img2img". Determines the type of pipeline to load.

    Returns:
        A configured StableDiffusionPipeline or StableDiffusionImg2ImgPipeline object, or None if initialization fails.
    """
    torch_dtype = torch.float16 if device_str == "cuda" else torch.float32
    pipeline_class = StableDiffusionImg2ImgPipeline if pipeline_type == "img2img" else StableDiffusionPipeline

    print(f"Attempting to load base model: {model_id_or_path} as {pipeline_type} pipeline.")

    pipe = None
    try:
        # Try loading as if it's a Hugging Face ID or a full local path recognized by from_pretrained
        pipe = pipeline_class.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        print(f"Successfully loaded model '{model_id_or_path}' using from_pretrained.")
    except (OSError, IOError) as e: # More specific exceptions for model loading issues
        print(f"Could not load model '{model_id_or_path}' directly with from_pretrained: {e}")
        if os.path.isdir(model_id_or_path):
            print(f"'{model_id_or_path}' is a directory. Attempting to load from local directory.")
            try:
                pipe = pipeline_class.from_pretrained(
                    model_id_or_path,
                    local_files_only=True, # Ensure it uses local files
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
    except Exception as e_general: # Catch other potential errors during from_pretrained
        print(f"An unexpected error occurred while loading model '{model_id_or_path}': {e_general}")
        return None

    pipe.to(device_str)
    # Using default scheduler, can be configured further if needed
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if lora_details:
        print(f"Loading {len(lora_details)} LoRA(s)...")
        for lora_info in lora_details:
            lora_path = lora_info.get('path')
            weight_name = lora_info.get('weight_name') # This is the filename of the LoRA
            adapter_name = lora_info.get('adapter_name') # Optional name for the adapter

            if not lora_path:
                print(f"Skipping LoRA due to missing path: {lora_info}")
                continue

            try:
                if weight_name: # If weight_name (filename) is given, lora_path is the directory
                    lora_dir = lora_path
                    actual_lora_file_path = os.path.join(lora_dir, weight_name)
                    if not os.path.exists(actual_lora_file_path):
                        print(f"LoRA file not found at {actual_lora_file_path}. Skipping.")
                        continue
                    print(f"Loading LoRA '{weight_name}' from directory '{lora_dir}'...")
                    pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=adapter_name)
                else: # If no weight_name, lora_path is the full path to the .safetensors file
                    if not os.path.exists(lora_path):
                        print(f"LoRA file not found at {lora_path}. Skipping.")
                        continue
                    # For load_lora_weights, the first arg is pretrained_model_name_or_path_or_dict
                    # which refers to the directory containing the LoRA, or a dict of tensors.
                    # The weight_name is the specific .safetensors file.
                    lora_file_dir = os.path.dirname(lora_path)
                    lora_filename = os.path.basename(lora_path)
                    if not lora_file_dir: # If it's just a filename, assume current directory
                        lora_file_dir = "."
                    print(f"Loading LoRA from full path '{lora_path}' (directory: '{lora_file_dir}', file: '{lora_filename}')...")
                    pipe.load_lora_weights(lora_file_dir, weight_name=lora_filename, adapter_name=adapter_name)

                print(f"Successfully loaded LoRA: {weight_name or lora_path}")
            except Exception as e:
                print(f"Could not load LoRA '{weight_name or lora_path}': {e}. Proceeding without this LoRA.")
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
    loras_to_load = [
        {'path': ".", 'weight_name': "pixel sprites.safetensors"} # Load from current dir
    ]
    # To test with a full path, you could use:
    # loras_to_load = [
    #    {'path': 'C:/path/to/your/loras/pixel_sprites.safetensors'} # No weight_name needed
    # ]
    pipe2 = initialize_diffusion_pipeline(
        "runwayml/stable-diffusion-v1-5",
        lora_details=loras_to_load,
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
    # pipe4 = initialize_diffusion_pipeline("path/to/your/local/sd/model/directory")
    # if pipe4:
    #     print("Test 4 Succeeded.")
    # else:
    #     print("Test 4 Failed.")
    print("\nPipeline.py testing finished.")