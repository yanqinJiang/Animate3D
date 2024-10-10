import torch
from pytorch3d.structures import Meshes, packed_to_list
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.renderer import TexturesVertex
import numpy as np

import os
import json

from plyfile import PlyData, PlyElement

import argparse

## mesh
def convert_to_textureVertex_with_average(meshes) -> TexturesVertex:

    textures_uv = meshes.textures
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    
    verts_colors_sum = torch.zeros_like(verts_packed)
    verts_count = torch.zeros(verts_packed.shape[0], 1, device=verts_packed.device)
 
    faces_textures = textures_uv.faces_verts_textures_packed()
    
    for i in range(3):
        verts_idx = faces_packed[:, i]
        verts_colors_sum.index_add_(0, verts_idx, faces_textures[:, i])
        verts_count.index_add_(0, verts_idx, torch.ones_like(verts_idx).unsqueeze(1).float())

    verts_colors_packed = verts_colors_sum / verts_count.clamp(min=1)

    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))

# mean value
def compute_edge_distances(meshes):
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    
    adj_list = {i: set() for i in range(verts_packed.shape[0])}
    
    for i in range(3):
        verts_idx_0 = faces_packed[:, i]
        verts_idx_1 = faces_packed[:, (i + 1) % 3]
        
        for v0, v1 in zip(verts_idx_0, verts_idx_1):
            adj_list[v0.item()].add(v1.item())
            adj_list[v1.item()].add(v0.item())
            
    num_verts = verts_packed.shape[0]
    mean_lengths_per_vert = torch.zeros((num_verts, 3), device=verts_packed.device) 
    count_per_vert = torch.zeros(num_verts, device=verts_packed.device)
    
    for v in range(num_verts):
        for adj in adj_list[v]:
            vec = verts_packed[adj] - verts_packed[v]
            mean_lengths_per_vert[v] += torch.abs(vec)
            count_per_vert[v] += 1
    
    nonzero_mask = count_per_vert > 0
    mean_lengths_per_vert[nonzero_mask] /= count_per_vert[nonzero_mask].unsqueeze(-1)
    
    return mean_lengths_per_vert
    
    
def find_and_save_connected_vertices(meshes, output_path):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    
    connected_vertices = {}
    
    for face in faces_packed:
        for i in range(3):
            v1 = face[i].item()
            v2 = face[(i+1)%3].item()
            
            distance = torch.norm(verts_packed[v1] - verts_packed[v2]).item()
            
            if v1 not in connected_vertices:
                connected_vertices[v1] = {}
            connected_vertices[v1][v2] = distance
            
            if v2 not in connected_vertices:
                connected_vertices[v2] = {}
            connected_vertices[v2][v1] = distance
    
    with open(output_path, 'w') as f:
        json.dump(connected_vertices, f, indent=2)
    
## gaussian
C0 = 0.28209479177387814

def construct_list_of_attributes():
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(3):
        l.append("f_dc_{}".format(i))
    for i in range(0):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(3):
        l.append("scale_{}".format(i))
    for i in range(4):
        l.append("rot_{}".format(i))
    return l
                

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

def convert_point_cloud_to_gaussian(verts_pos, verts_colors, max_lengths_per_vert, output_path):
    
    xyz = verts_pos
    normals = np.zeros_like(xyz)
    f_dc = RGB2SH(verts_colors)
    f_rest = np.zeros((xyz.shape[0], 0))
    
    scale = np.log(max_lengths_per_vert)
    rotation = np.zeros((xyz.shape[0], 4))
    rotation[:, 0] = 1.
    
    opacities = inverse_sigmoid(np.ones((xyz.shape[0], 1))-0.00001)
    
    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes()
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)
    
    return

def main():
    parser = argparse.ArgumentParser(description="Convert OBJ to Gaussian point cloud and extract connected vertices info.")
    parser.add_argument("--input_obj", help="Path to input OBJ file")
    parser.add_argument("--output_dir", help="Directory to save output files")
    parser.add_argument("--output_name", help="Base name for output files (without extension)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    output_ply_path = os.path.join(args.output_dir, f"{args.output_name}.ply")
    output_connected_vertices_path = os.path.join(args.output_dir, f"{args.output_name}.json")

    # Load OBJ file
    meshes = load_objs_as_meshes([args.input_obj], device="cuda")

    # Convert to colored vertices
    if isinstance(meshes.textures, TexturesVertex):
        colored_verts = meshes.textures
    else:
        colored_verts = convert_to_textureVertex_with_average(meshes)

    # Get vertex positions and colors
    verts_pos = meshes.verts_packed().cpu().numpy()
    verts_colors = colored_verts.verts_features_packed().cpu().numpy()

    # Compute maximum edge lengths
    max_lengths_per_vert = compute_edge_distances(meshes).cpu().numpy()

    # Convert to Gaussian point cloud and save as PLY
    convert_point_cloud_to_gaussian(verts_pos, verts_colors, max_lengths_per_vert / 1.1, output_ply_path)
    print(f"Gaussian point cloud saved to {output_ply_path}")

    # Find and save connected vertices information
    find_and_save_connected_vertices(meshes, output_connected_vertices_path)
    print(f"Connected vertices information saved to {output_connected_vertices_path}")

if __name__ == "__main__":
    main()
