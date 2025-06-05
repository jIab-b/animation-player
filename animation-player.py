import pygame
import os
import argparse
import time

def play_animation(folder_path, fps=10):
    """
    Plays an animation from a folder of sequentially named PNG images.
    Args:
        folder_path (str): Path to the folder containing image frames (0.png, 1.png, ...).
        fps (int): Frames per second for the animation.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    pygame.init()

    frames = []
    i = 0
    while True:
        frame_path = os.path.join(folder_path, f"{i}.png")
        if os.path.exists(frame_path):
            try:
                frames.append(pygame.image.load(frame_path))
                i += 1
            except pygame.error as e:
                print(f"Could not load image {frame_path}: {e}")
                # Decide if you want to skip or stop
                break # Stop if an image in sequence is unloadable
        else:
            break # No more frames in sequence

    if not frames:
        print(f"No valid image frames found in {folder_path} (e.g., 0.png, 1.png, ...)")
        pygame.quit()
        return

    # Determine screen size from the first frame
    screen_width, screen_height = frames[0].get_size()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Animation Player")

    clock = pygame.time.Clock()
    running = True
    current_frame_index = 0
    num_frames = len(frames)

    print(f"Playing animation from {folder_path} ({num_frames} frames) at {fps} FPS.")
    print("Press ESC or close the window to stop.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.blit(frames[current_frame_index], (0, 0))
        pygame.display.flip()

        current_frame_index = (current_frame_index + 1) % num_frames
        clock.tick(fps)

    pygame.quit()
    print("Animation stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play an animation from a folder of PNG images.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing animation frames (e.g., 'animate_xxxxxxxx').")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the animation. Default is 10.")
    
    args = parser.parse_args()
    
    play_animation(args.input_folder, args.fps)