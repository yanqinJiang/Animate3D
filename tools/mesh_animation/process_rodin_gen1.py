import shutil
import os
import argparse

def process_file(source_path, save_path):
    # Path to the template MTL file
    template_mtl_path = "tools/mesh_animation/templates/rodin_gen1/base.mtl"

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Copy the template MTL file to the save directory
    target_mtl_path = os.path.join(save_path, "base.mtl")
    shutil.copy(template_mtl_path, target_mtl_path)

    # Path to the source OBJ file
    obj_path = os.path.join(source_path, "base.obj")

    # Path to the new OBJ file in the save directory
    new_obj_path = os.path.join(save_path, "base.obj")

    # Read the contents of the source OBJ file
    with open(obj_path, "r") as file_to_read:
        lines = file_to_read.readlines()

    # Write the modified contents to the new OBJ file
    with open(new_obj_path, "w+") as file_to_write:
        file_to_write.write(lines[0])
        file_to_write.write("usemtl Material\n")
        
        for line in lines[1:]:
            file_to_write.write(line)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a single OBJ file.")
    parser.add_argument("--source_path", help="Path to the source directory containing the base.obj file")
    parser.add_argument("--save_path", help="Path to save the processed files")

    # Parse arguments
    args = parser.parse_args()

    # Process the file
    process_file(args.source_path, args.save_path)

if __name__ == "__main__":
    main()
