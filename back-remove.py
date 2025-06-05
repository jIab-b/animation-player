import argparse
from rembg import remove
from PIL import Image
import os

def remove_background(input_path, output_path):
    """
    Removes the background from an image using rembg.
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output image with background removed.
    """
    try:
        with open(input_path, 'rb') as i:
            input_data = i.read()
            output_data = remove(input_data)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        with open(output_path, 'wb') as o:
            o.write(output_data)
        print(f"Background removed. Output saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have 'rembg' installed (e.g., 'pip install rembg').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("--output_image", type=str, help="Optional: Path to save the output image. Defaults to '<input_filename>-tp.<ext>' in the same directory.")
    
    args = parser.parse_args()

    output_path = args.output_image
    if not output_path:
        input_dir = os.path.dirname(args.input_image)
        input_filename, input_ext = os.path.splitext(os.path.basename(args.input_image))
        output_filename = f"{input_filename}-tp{input_ext}"
        output_path = os.path.join(input_dir, output_filename)
        if not input_dir: # Handle case where input is in current directory
            output_path = output_filename
            
    remove_background(args.input_image, output_path)