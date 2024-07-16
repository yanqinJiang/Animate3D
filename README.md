# Animate3D: Animating Any 3D Model with Multi-view Video Diffusion
Yanqin Jiang<sup>1*</sup>, Chaohui Yu<sup>2*</sup>, Chenjie Cao<sup>2</sup>, Fan Wang<sup>2</sup>, Weiming Hu<sup>1</sup>, Jin Gao<sup>1</sup><br>

<sup>1</sup>CASIA, <sup>2</sup>DAMO Academy, Alibaba Group

| [Project Page](https://animate3d.github.io/) | [arXiv]() | [Paper]() | [Video](https://www.youtube.com/watch?v=qkaeeGzLnY8) | Data |

![Demo GIF](https://github.com/animate3d/animate3d.github.io/blob/main/assets/bg.gif)

# Abstract
Recent advances in 4D generation mainly focus on generating 4D content by distilling pre-trained text or single-view image-conditioned models. 
It is inconvenient for them to take advantage of various off-the-shelf 3D assets with multi-view attributes, and their results suffer from spatiotemporal inconsistency owing to the inherent ambiguity in the supervision signals.
In this work, we present Animate3D, a novel framework for animating any static 3D model.
The core idea is two-fold: 1) We propose a novel multi-view video diffusion model (MV-VDM) conditioned on multi-view renderings of the static 3D object, which is trained on our presented large-scale multi-view video dataset (MV-Video). 2) Based on MV-VDM, we introduce a framework combining reconstruction and 4D Score Distillation Sampling (4D-SDS) to leverage the multi-view video diffusion priors for animating 3D objects.
Specifically, for MV-VDM, we design a new spatiotemporal attention module to enhance spatial and temporal consistency by integrating 3D and video diffusion models. 
Additionally, we leverage the static 3D model's multi-view renderings as conditions to preserve its identity.
For animating 3D models, an effective two-stage pipeline is proposed: we first reconstruct motions directly from generated multi-view videos, followed by the introduced 4D-SDS to refine both appearance and motion.
Qualitative and quantitative experiments demonstrate that Animate3D significantly outperforms previous approaches.
Data, code, and models will be open-released.

# News
[**2024.07.17**] The paper of Consistent4D is avaliable at [arXiv]().

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


