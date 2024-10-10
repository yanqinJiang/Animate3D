import os
from PIL import Image

from PIL import Image
import argparse

def split_gif_frames(gif_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with Image.open(gif_path) as img:

        frame_count = 0
        while True:
            try:
                img.seek(frame_count)
                frame_count += 1
            except EOFError:
                break 
        
        img.seek(0)

        frame_num = 0
        while True:
            try:
                img.seek(frame_num)
            except EOFError:
                break 

            temp_path = os.path.join(output_dir, f"temp_frame_{frame_num}.png")
            img.save(temp_path)
            
            with Image.open(temp_path) as temp_img:

                width, height = temp_img.size
                square_size = width // 4
                
                for i in range(4):
     
                    left = i * square_size
                    upper = 0
                    right = left + square_size
                    lower = height

     
                    square_img = temp_img.crop((left, upper, right, lower))
                    square_img.save(os.path.join(output_dir, f"{i * frame_count + frame_num}.png"))

            os.remove(temp_path)
            
            frame_num += 1

def main():
    parser = argparse.ArgumentParser(description="Split GIF frames into separate images.")
    parser.add_argument('--gif_path', type=str, required=True, help="Path to the input GIF file")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder")
    
    args = parser.parse_args()

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    gif_path = args.gif_path
    output_dir = os.path.join(output_folder, os.path.basename(gif_path)[:-4])
    
    split_gif_frames(gif_path, output_dir)
    
    # if not os.path.exists(output_dir):
    #     split_gif_frames(gif_path, output_dir)
    # else:
    #     print(f"Output directory {output_dir} already exists. Skipping processing.")

if __name__ == "__main__":
    main()
