import bpy
import numpy as np
import os
import argparse
import sys

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_material(name):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    return material

def add_texture_to_material(material, texture_name, texture_type, texture_dir):
    nodes = material.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    
    texture_image = bpy.data.images.load(os.path.join(texture_dir, f"texture_{texture_name}.png"))
    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = texture_image
    
    if texture_type == 'diffuse':
        material.node_tree.links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    elif texture_type == 'metallic':
        material.node_tree.links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Metallic'])
    elif texture_type == 'roughness':
        material.node_tree.links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Roughness'])
    elif texture_type == 'normal':
        normal_map = nodes.new('ShaderNodeNormalMap')
        material.node_tree.links.new(texture_node.outputs['Color'], normal_map.inputs['Color'])
        material.node_tree.links.new(normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])

def main():
    parser = argparse.ArgumentParser(description="Process OBJ file and create animated FBX.")
    parser.add_argument("--obj_dir", help="Directory containing base.obj and textures")
    parser.add_argument("--npy_dir", help="Directory containing vertex animation NPY files")
    parser.add_argument("--output_path", help="Output path for FBX file")
    parser.add_argument("--theta_x_degree", type=float, default=90.0, help="transform x degree")
    parser.add_argument("--theta_z_degree", type=float, default=90.0, help="transform z degree")
    parser.add_argument("--scale_factor", type=float, default=0.76, help="scale factor")
    
    args = parser.parse_args()

    obj_path = os.path.join(args.obj_dir, "base.obj")
    texture_dir = args.obj_dir
    npy_dir = args.npy_dir
    output_path = args.output_path

    theta_x_degree = args.theta_x_degree
    theta_z_degree = args.theta_z_degree
    scale_factor = args.scale_factor
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clear_scene()
    bpy.ops.wm.obj_import(filepath=obj_path, use_split_objects=False, use_split_groups=False)
    obj = bpy.context.selected_objects[0]

    material = create_material("ObjectMaterial")
    obj.data.materials.append(material)
    for tex_type in ["diffuse", "metallic", "roughness", "normal"]:
        add_texture_to_material(material, tex_type, tex_type, texture_dir)

    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')], key=lambda x: int(x.split('.')[0]))
    mesh = obj.data

    if obj.data.shape_keys:
        obj.shape_key_clear()

    bpy.context.view_layer.objects.active = obj
    basis_shape = obj.shape_key_add(name="Basis")
    basis_shape.interpolation = 'KEY_LINEAR'

    # prepare transformation matrix
    theta_x = np.deg2rad(theta_x_degree)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    theta_z = np.deg2rad(theta_z_degree)
    rotation_matrix_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    rotation_matrix = rotation_matrix_z @ rotation_matrix_x
    rotation_matrix_inv = rotation_matrix.T
    
    for frame, npy_file in enumerate(npy_files):
        # vertex_positions = np.load(os.path.join(npy_dir, npy_file))
        mesh_offset = np.load(os.path.join(npy_dir, npy_file))
        new_mesh_offset = mesh_offset / scale_factor
        new_mesh_offset = (rotation_matrix_inv @ new_mesh_offset.T).T
        vertex_positions = new_mesh_offset
        
        shape_key = obj.shape_key_add(name=f"Key_{frame:03d}", from_mix=False)
        shape_key.interpolation = 'KEY_LINEAR'
        for i, vertex in enumerate(mesh.vertices):
            shape_key.data[i].co = vertex_positions[i]
        
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame)
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path="value", frame=frame+1)
        
        if frame > 0:
            prev_shape_key = obj.data.shape_keys.key_blocks[f"Key_{frame-1:03d}"]
            prev_shape_key.value = 1.0
            prev_shape_key.keyframe_insert(data_path="value", frame=frame)
            prev_shape_key.value = 0.0
            prev_shape_key.keyframe_insert(data_path="value", frame=frame+1)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(npy_files)

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
        use_mesh_modifiers=True,
        use_mesh_edges=True,
        use_tspace=True,
        use_custom_props=True,
        use_active_collection=False,
    )
    print("Animation exported successfully!")

if __name__ == "__main__":
    main()
