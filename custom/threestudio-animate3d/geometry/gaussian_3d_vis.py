#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple, Sequence, Collection

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

# from custom.threestudio_3dgs.geometry.gaussian_base import GaussianBaseModel
import sys
GaussianBaseModel = getattr(sys.modules["threestudio-3dgs.geometry.gaussian_base"], 'GaussianBaseModel')

from threestudio.models.networks import get_mlp

from .utils import build_rotation_np, extract_rotation_scipy
import itertools


@threestudio.register("gaussian-splatting-3d-vis")
class Gaussian3DVisModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        # load_ply config
        load_ply_cfg: dict = field(default_factory=dict)

    # modified from gussian_io: load_ply func
    def load_ply(self, path) -> None:
        
        theta_x_degree = self.cfg.load_ply_cfg.rot_x_degree
        theta_z_degree = self.cfg.load_ply_cfg.rot_z_degree
        scale_factor = self.cfg.load_ply_cfg.scale_factor

        theta_x = np.deg2rad(theta_x_degree)
        # 创建旋转矩阵
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
        
        # load ply
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        # xyz: rotate and scale
        xyz = (rotation_matrix @ xyz.T).T
        xyz = xyz * scale_factor

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.max_sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # scaling: rotate and scale
        scales = np.log(np.exp(scales) * scale_factor)

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # rotation: rotate, WITHOUT this step leading to blury renderings!!
        new_rotation = build_rotation_np(rots)
        # new_rotation = np.dot(new_rotation, rotation_matrix_x_90.T) # wrong
        new_rotation = rotation_matrix @ new_rotation
        rots = extract_rotation_scipy(new_rotation)


        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        if self.max_sh_degree > 0:
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
        else:
            self._features_rest = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda")[:, :, 1:]
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
        self._opacity = nn.Parameter(
            torch.tensor(opacities.copy(), dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        ) # note: numpy stride
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree