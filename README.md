# Animate3D: Animating Any 3D Model with Multi-view Video Diffusion
Yanqin Jiang<sup>1*</sup>, Chaohui Yu<sup>2*</sup>, Chenjie Cao<sup>2</sup>, Fan Wang<sup>2</sup>, Weiming Hu<sup>1</sup>, Jin Gao<sup>1</sup><br>

<sup>1</sup>CASIA, <sup>2</sup>DAMO Academy, Alibaba Group

| [Project Page](https://animate3d.github.io/) | [arXiv](https://arxiv.org/abs/2407.11398) | Paper | [Video](https://www.youtube.com/watch?v=qkaeeGzLnY8) | Training Data | [Test Data](https://drive.google.com/file/d/1iFSuCAwWBVzlLCQH32yoikz8M2qBJ8rP/view?usp=sharing) |

![Demo GIF](https://github.com/animate3d/animate3d.github.io/blob/main/assets/bg.gif)

![Demo GIF](https://github.com/animate3d/animate3d.github.io/blob/main/assets/mesh_demo_resized.gif)

**This repository will be under long-term maintenance. Feel free to open issues or submit pull requests.**

# Abstract
Recent advances in 4D generation mainly focus on generating 4D content by distilling pre-trained text or single-view image-conditioned models. 
It is inconvenient for them to take advantage of various off-the-shelf 3D assets with multi-view attributes, and their results suffer from spatiotemporal inconsistency owing to the inherent ambiguity in the supervision signals.
In this work, we present Animate3D, a novel framework for animating any static 3D model.
The core idea is two-fold: 1) We propose a novel multi-view video diffusion model (MV-VDM) conditioned on multi-view renderings of the static 3D object, which is trained on our presented large-scale multi-view video dataset (MV-Video). 2) Based on MV-VDM, we introduce a framework combining reconstruction and 4D Score Distillation Sampling (4D-SDS) to leverage the multi-view video diffusion priors for animating 3D objects.
Specifically, for MV-VDM, we design a new spatiotemporal attention module to enhance spatial and temporal consistency by integrating 3D and video diffusion models. 
Additionally, we leverage the static 3D model's multi-view renderings as conditions to preserve its identity.
For animating 3D models, an effective two-stage pipeline is proposed: we first reconstruct motions directly from generated multi-view videos, followed by the introduced 4D-SDS to refine both appearance and motion. Benefiting from accurate motion learning, we could achieve straightforward mesh animation.
Qualitative and quantitative experiments demonstrate that Animate3D significantly outperforms previous approaches.
Data, code, and models will be open-released.

# News
[**2024.10.10**] Codes and pretrained models of Animate3D are released! Dataset will be released in one week. <br>
[**2024.09.26**] Animate3D is accepted by [NeurIPS 2024](https://neurips.cc/)! Thanks for all! : ) <br>
[**2024.09.10**] 😄❤️❤️ **Animate3D introduces an exciting new feature: Mesh Animation. Mesh animation requires only 15 minutes in total.** We uploaded mesh animations to our project page one month ago, and now we provide **technical details** in [updated paper](https://arxiv.org/abs/2407.11398). **Examples of animated files in FBX format can be found [here](https://drive.google.com/file/d/1RpOhNA8c8Bm-ShCInHixH6Q-AR1ZTARN/view?usp=drive_link), ready for import into standard 3D software such as Blender.** <br>
[**2024.07.17**] The paper of Animate3D is avaliable at [arXiv](https://arxiv.org/abs/2407.11398)! We achieve impressing results, and we present high-resolution video on our project page : )

# Installation
Please refer to [install.md](docs/install.md).

# Pretrained Models and Example Data
Download pretrained [animate3d_motion_modules.ckpt](https://huggingface.co/yanqinJiang/animate3d/blob/main/animate3d_motion_modules.ckpt) and put it under `pretrained_models` folder. Download the [test data](https://drive.google.com/file/d/1iFSuCAwWBVzlLCQH32yoikz8M2qBJ8rP/view?usp=sharing) and unzip it. You will expect to see:
```
Animate3D/
├── data/
│   ├── animate3d/
│   └── vdm/
├── pretrained_models/
│   └── animate3d_motion_modules.ckpt
...
```
Note that we convert [MVDream](https://github.com/bytedance/MVDream/) to diffusers and the converted weight is uploaded to [Huggingface](https://huggingface.co/yanqinJiang/mvdream-sd1.5-diffusers). The code is expected to download this model automatically, but if not, you could manually download it and replace `"yanqinJiang/mvdream-sd1.5-diffusers"` with `"/path/to/your/downloaded/folder/"` in configs.

# MV-VDM: Training and Inference
Please refer to [mv-vdm.md](docs/mv-vdm.md).
# Animating 3D Model
We support animating both **Mesh** and **Gaussian Splatting** models. Step-by-step instructions are provided as bellow.
## Mesh (Experimental but Recommended)
Currently we only support animating mesh in `.obj` format. Normally, we would expect the mesh file is organized as follows:
```bash
object_name/
├── base.mtl
├── base.obj
├── texture_diffuse.png
├── texture_metallic.png
├── texture_normal.png
├── texture_pbr.png
└── texture_roughness.png
```
Please see `data/animate3d/mesh/obj_file/dragon_head` for an example. Sometimes ``base.mtl`` might be missing, for example, mesh models downloaded from [Rodin-Gen1](https://hyperhuman.deemos.com/rodin) usually do not have this file. So we provide a simple script, [process_rodin_gen1.py](tools/mesh_animation/process_rodin_gen1.py), to handle such cases. Use it with following command:
```bash
python tools/mesh_animation/process_rodin_gen1.py \
    --source_path /path/to/your/model/folder \
    --save_path /path/to/save/folder
```
Once the mesh model is prepared, run the following command to do the animation. Here we take a `dragon_head` model as example:
```bash
# Step1 mesh2gaussian: we provide a simple script to extract coarse gaussian model from mesh object.  This script typically yields beeter results when applied to generated mesh objects featuring evenly distributed vertices and faces.
# results saved to data/animate3d/mesh/converted_gaussian/
python tools/mesh_animation/mesh2gaussian.py \
    --input_obj data/animate3d/mesh/obj_file/dragon_head/base.obj \
    --output_dir data/animate3d/mesh/converted_gaussian \
    --output_name dragon_head

# Step2 rendering 4-views of the gaussian object: the renders serve as the image condition for mv-vdm. Note that the coordinate of the mesh and the pre-defined gaussian system might not be the same, so you should manually check the system.geometry.load_ply_cfg !!! The load_ply_cfg used here is set for the given example (objects from Rodin-Gen1 could use this cfg too).
# results saved to outputs/animate3d/static_vis/
python launch.py \
    --config custom/threestudio-animate3d/configs/visualize_four_view_static.yaml \
    --test \
    --gpu 0 \
    name="static_vis" \
    tag="dragon_head" \
    system.prompt_processor.prompt="visualize" \
    system.geometry.geometry_convert_from="data/animate3d/mesh/converted_gaussian/dragon_head.ply" \
    system.geometry.load_ply_cfg.rot_x_degree=90. \
    system.geometry.load_ply_cfg.rot_z_degree=90. \
    system.geometry.load_ply_cfg.scale_factor=0.76 

# Step3 mv-vdm inference, this shoud generate one animations in gif format.
# results saved to outputs/animate3d/animation_gif/
python inference.py \
    --config "configs/inference/inference.yaml" \
    --pretrained_unet_path "pretrained_models/animate3d_motion_modules.ckpt" \
    --W 256 \
    --H 256 \
    --L 16 \
    --N 4 \
    --ip_image_root "outputs/animate3d/static_vis/dragon_head/save/images" \
    --ip_image_name "" \
    --prompt "a wooden dragon head is roaring" \
    --save_name "animate3d/animation_gif"

# Step4 split the gif file to images and segment the foreground object
# results saved to outputs/animate3d/animation_images/
python tools/split_gif.py \
    --gif_path outputs/animate3d/animation_gif/0-a-wooden-dragon-head-is-roaring.gif \
    --output_folder outputs/animate3d/animation_images
# results saved to data/animate3d/mesh/tracking_rgba_images/
cd tools/tracking_anything
python custom_inference.py \
    --folder_path ../../outputs/animate3d/animation_images/0-a-wooden-dragon-head-is-roaring \
    --save_path ../../data/animate3d/mesh/tracking_rgba_images/0-a-wooden-dragon-head-is-roaring \
    --template_mask_folder ../../outputs/animate3d/static_vis/dragon_head/save/images

# Step5 Animate Mesh!
# results saved to outputs/animate3d/mesh
cd ../..
python launch.py \
    --config custom/threestudio-animate3d/configs/mesh_animation_frame_16.yaml  \
    --train \
    --gpu 0 \
    tag="dragon_head" \
    system.prompt_processor.prompt="A wooden dragon head is roaring." \
    system.geometry.geometry_convert_from="data/animate3d/mesh/converted_gaussian/dragon_head.ply" \
    system.geometry.load_ply_cfg.rot_x_degree=90. \
    system.geometry.load_ply_cfg.rot_z_degree=90. \
    system.geometry.load_ply_cfg.scale_factor=0.76 \
    data.image_root="data/animate3d/mesh/tracking_rgba_images/0-a-wooden-dragon-head-is-roaring" \
    system.connected_vertices_info_path="data/animate3d/mesh/converted_gaussian/dragon_head.json" 

# Step6 Visualize the mesh and save gaussian trajectory
# results saved to outputs/animate3d/mesh_vis
python launch.py \
    --config custom/threestudio-animate3d/configs/visualize_four_view_frame_16.yaml  \
    --test \
    --gpu 0 \
    name="mesh_vis" \
    tag="dragon_head" \
    system.prompt_processor.prompt="visualize" \
    resume="outputs/animate3d/mesh/dragon_head/ckpts/epoch=0-step=800.ckpt" \
    system.save_gaussian_trajectory=True \

# Step7 export animated mesh in fbx format
# results saved to outputs/animate3d/mesh_vis
python tools/mesh_animation/export_animated_mesh.py \
    --obj_dir data/animate3d/mesh/obj_file/dragon_head \
    --npy_dir outputs/animate3d/mesh_vis/dragon_head/save/mesh_trajectory \
    --output_path outputs/animate3d/mesh_vis/dragon_head/save/animate3d_model.fbx \
    --theta_x_degree 90. \
    --theta_z_degree 90. \
    --scale_factor 0.76 
```

## Gaussian Splatting
We have provided the pre-trained static gaussian and animation videos used in the paper (in rgba image format) in `data/animate3d/testset`. Use the following command to reproduce the results in our paper:
```bash
gpu=0
object_name=butterfly # change it to other name
prompt="A glowing butterfly is flying."

# Step1 motion reconstruction
# results saved to outputs/animate3d/recon/
recon_config=motion_recon_frame_16
python launch.py --config custom/threestudio-animate3d/configs/$recon_config.yaml  \
    --train \
    --gpu $gpu \
    tag=$object_name \
    system.prompt_processor.prompt="${prompt}" \
    system.geometry.geometry_convert_from="data/animate3d/testset/pretrained_gaussian/${object_name}.ply" \
    data.image_root="data/animate3d/testset/tracking_rgba_images/${object_name}" \

# Step2 visualize the recon results
# results saved to outputs/animate3d/recon_vis
vis_config=visualize_four_view_frame_16
python launch.py --config custom/threestudio-animate3d/configs/$vis_config.yaml  \
    --test \
    --gpu $gpu \
    name="recon_vis" \
    tag=$object_name \
    system.prompt_processor.prompt="${prompt}" \
    system.geometry.geometry_convert_from="data/animate3d/testset/pretrained_gaussian/${object_name}.ply" \
    data.image_root="data/animate3d/testset/tracking_rgba_images/${object_name}" \
    resume="outputs/animate3d/recon/${object_name}/ckpts/epoch=0-step=800.ckpt" \

# Step3 refine
# results saved to outputs/animate3d/refine
refine_config=refine_frame_16
python launch.py --config custom/threestudio-animate3d/configs/$refine_config.yaml  \
    --train \
    --gpu $gpu \
    tag=${object_name} \
    system.prompt_processor.prompt="${prompt}" \
    system.geometry.geometry_convert_from="data/animate3d/testset/pretrained_gaussian/${object_name}.ply" \
    data.image_root="outputs/animate3d/recon_vis/${object_name}/save/images" \
    system.weights="outputs/animate3d/recon/${object_name}/ckpts/epoch=0-step=800.ckpt"

# Step4 render images for testing
# results saved to outputs/animate3d/refine_testset
vis_config=visualize_testset_frame_16
python launch.py --config custom/threestudio-animate3d/configs/$vis_config.yaml  \
    --test \
    --gpu $gpu \
    name="refine_testset" \
    tag=$object_name \
    system.prompt_processor.prompt="${prompt}" \
    system.geometry.geometry_convert_from="data/animate3d/testset/pretrained_gaussian/${object_name}.ply" \
    data.image_root="data/animate3d/testset/tracking_rgba_images/${object_name}" \
    resume="outputs/animate3d/refine/${object_name}/ckpts/epoch=0-step=200.ckpt" \

```
# TODO
- [x] Release training and inference code for MV-VDM.
- [x] Support Gaussian Splatting Animation.
- [x] Support Mesh Animation
- [ ] Support multi-object animation
- [ ] Support more mesh format

# Tips
* **Code structure**: The 3D Animation code is implemented as a plugin for threestudio. For your convenience, we have included the base code of threestudio (i.e, folder `threestudio`, `extern`) in our repository. If you already have a threestudio setup, you can simply copy the `custom/threestudio-animate3d` directory into your existing threestudio codebase.
* **Motion diversity**: We use [FreeInit](https://github.com/TianxingWu/FreeInit) to improve the temporal consistency of the generated video. The iterative factor is set to 3 by default (see `freeinit_num_iters` in `configs/inference/inference.yaml`). Increasing the iterative factor will improve the temporal consistency but decrease the motion diversity, and vice versa. So you can choose the appropriate factor by yourself.
* **Motion amplitude**: There is a trade-off between the amplitude of the motion and the fedility of the appearance. You can finetune the `system.loss.lambda_arap` to do the trade-off. As the value increases, the motion amplitude decreases, while the appearance fidelity improves. **Here we set it to a relatively higher value to improve the success rate. Feel free to adjust it according to your needs.**
* **Text alignment**: Sometimes the generated motion might not align with the text prompt, and that is because the video caption used for training MV-VDM is generated by video caption models, and it's not always accurate. Try to use a different seed by changing `seed` option in `configs/inference/inference.yaml`.
* **Training iteration**: Setting training iteration to a larger value sometimes helps with natural motion reconstruction.
* **Multi-object animation**: Currently, our system **does not support multi-object animation**. As a result, performance in scenarios involving multiple objects cannot be guaranteed. We'll support it in the future.

# Acknowledgement
The code of Animate3D is based on [AnimateDiff](https://github.com/guoyww/AnimateDiff) and [threestudio](https://github.com/threestudio-project/threestudio). We thank the authors for their effort in building such great codebases. We also take inspiration from [Consistent4D](https://github.com/yanqinJiang/Consistent4D), [STAG4D](https://github.com/zeng-yifei/STAG4D) and [Track-Anything](https://github.com/gaomingqi/Track-Anything) when building our framework. Thanks for their great works.

# Citation
```bibtex
@article{
jiang2024animate3d,
title={Animate3D: Animating Any 3D Model with Multi-view Video Diffusion},
author={Yanqin Jiang and Chaohui Yu and Chenjie Cao and Fan Wang and Weiming Hu and Jin Gao},
booktitle={arXiv},
year={2024},
}
```