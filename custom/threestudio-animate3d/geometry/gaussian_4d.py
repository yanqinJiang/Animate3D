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

from .utils import build_rotation_np, extract_rotation_scipy, build_rotation, extract_rotation_torch, euler_angles_to_rotation_matrix
import itertools

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


@threestudio.register("gaussian-splatting-4d")
class Gaussian4DModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        # kplanes
        grid_size: Tuple[Tuple[int, int, int, int]] = field(default_factory=lambda: ((50, 50, 50, 4), (100, 100, 100, 16)))
        n_input_dims: int = 4
        n_grid_dims: int = 16
        # for deformnet: delta xyz, delta rot, delta scale
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            }
        )

        # global rot and trans
        use_global_trans: bool = False
        
        delta_xyz_network_lr: float = 0.1
        delta_scaling_network_lr: float = 0.1
        delta_rot_network_lr: float = 0.1
        
        global_trans_lr: float = 0.1
        grid_lr: float = 0.1
        # TODO: load_ply is temporally overwite
        # load_ply config
        load_ply_cfg: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:

        # kplans
        self.grids = nn.ModuleList()
        self.feat_dims = 0
        for idx, resolution in enumerate(self.cfg.grid_size):
           
            gp = self.init_grid_param(
                grid_nd=2,
                in_dim=self.cfg.n_input_dims,
                out_dim=self.cfg.n_grid_dims,
                reso=resolution,
            )
            self.grids.append(gp)
            
            self.feat_dims += gp[-1].shape[1]

        # deform
        self.delta_xyz_network = get_mlp(
            self.feat_dims, 3, self.cfg.mlp_network_config
        )
        self.delta_rot_network = get_mlp(
            self.feat_dims, 4, self.cfg.mlp_network_config
        )
        self.delta_scaling_network = get_mlp(
            self.feat_dims, 3, self.cfg.mlp_network_config
        )

        if self.cfg.use_global_trans:
            self.global_rot_network = get_mlp(
                self.feat_dims, 3, self.cfg.mlp_network_config
            )
            
            self.global_trans_network = get_mlp(
                self.feat_dims, 3, self.cfg.mlp_network_config
            )
            
            self.global_rot_trans_activation = torch.nn.Sigmoid()
            
            # zero init
            nn.init.zeros_(self.global_rot_network.layers[2].weight)
            nn.init.zeros_(self.global_trans_network.layers[2].weight)
            
        # zero init
        nn.init.zeros_(self.delta_xyz_network.layers[2].weight)
        nn.init.zeros_(self.delta_rot_network.layers[2].weight)
        nn.init.zeros_(self.delta_scaling_network.layers[2].weight)

        super().configure()

    def init_grid_param(
        self,
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time_planes = in_dim == 4
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            ))
            if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)

        return grid_coefs

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
        new_rotation = rotation_matrix @ new_rotation
        rots = extract_rotation_scipy(new_rotation)
        
        delattr(self, '_xyz')
        self.register_buffer(
            "_xyz", torch.tensor(xyz, dtype=torch.float, device="cuda")
        )
        
        delattr(self, '_features_dc')
        self.register_buffer(
            "_features_dc", 
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(False)
        )
        
        delattr(self, '_features_rest')
        if self.max_sh_degree > 0:
            self.register_buffer(
                "_features_rest",
                torch.tensor(features_extra, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(False)
            )
        else:
            self.register_buffer(
                "_features_rest",
                torch.tensor(features_dc, dtype=torch.float, device="cuda")[:, :, 1:]
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(False)
            )
            
        delattr(self, '_opacity')
        self.register_buffer(
            "_opacity", torch.tensor(opacities.copy(), dtype=torch.float, device="cuda")
        )
        
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree


    def get_xyz_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "delta_xyz" in name:
                parameter_list.append(param)
        return parameter_list

    def get_rot_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "delta_rot" in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_scaling_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "delta_scaling" in name:
                parameter_list.append(param)
        return parameter_list
    
    
    def get_global_trans_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "global" in name:
                parameter_list.append(param)
        return parameter_list
    
    

    def get_grid_parameters(self):
            return list(self.grids.parameters()) 

    # modified from gaussian_base: training_setup func
    def training_setup(self):
        
        training_args = self.cfg
       
        l = []
        l = l + [
            {
                "params": list(self.get_xyz_mlp_parameters()),
                "lr": C(training_args.delta_xyz_network_lr, 0, 0),
                "name": "delta_xyz_network",
            },
            {
                "params": list(self.get_rot_mlp_parameters()),
                "lr": C(training_args.delta_rot_network_lr, 0, 0),
                "name": "delta_rot_network",
            },
            {
                "params": list(self.get_scaling_mlp_parameters()),
                "lr": C(training_args.delta_scaling_network_lr, 0, 0),
                "name": "delta_scaling_network",
            },
            {
                "params": list(self.get_grid_parameters()),
                "lr": C(training_args.grid_lr, 0, 0),
                "name": "grid",
            },
        ]

        if self.cfg.use_global_trans:
            l.append(
                {
                    "params": list(self.get_global_trans_parameters()),
                    "lr": C(training_args.global_trans_lr, 0, 0),
                    "name": "global_trans",
                }
            )
            
        if self.cfg.pred_normal:
            l.append(
                {
                    "params": [self._normal],
                    "lr": C(training_args.normal_lr, 0, 0),
                    "name": "normal",
                },
            )

        self.optimize_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def update_learning_rate(self, iteration):
    
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "xyz":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "scaling":
                param_group["lr"] = C(
                    self.cfg.scaling_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "f_dc":
                param_group["lr"] = C(
                    self.cfg.feature_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "f_rest":
                param_group["lr"] = (
                    C(self.cfg.feature_lr, 0, iteration, interpolation="exp") / 20.0
                )
            if param_group["name"] == "opacity":
                param_group["lr"] = C(
                    self.cfg.opacity_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "rotation":
                param_group["lr"] = C(
                    self.cfg.rotation_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "normal":
                param_group["lr"] = C(
                    self.cfg.normal_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "delta_xyz_network":
                param_group["lr"] = C(
                    self.cfg.delta_xyz_network_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "delta_rot_network":
                param_group["lr"] = C(
                    self.cfg.delta_rot_network_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "delta_scaling_network":
                param_group["lr"] = C(
                    self.cfg.delta_scaling_network_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "global_trans":
                param_group["lr"] = C(
                    self.cfg.global_trans_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "grid":
                param_group["lr"] = C(
                    self.cfg.grid_lr, 0, iteration, interpolation="exp"
                )

        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def interpolate_ms_features(self,
                                pts: torch.Tensor,
                                ms_grids: Collection[Iterable[nn.Module]],
                                grid_dimensions: int = 2,
                                concat_features: bool = True,
                                num_levels: Optional[int] = None,
                                ) -> torch.Tensor:
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
        if num_levels is None:
            num_levels = len(ms_grids)
        multi_scale_interp = [] if concat_features else 0.
        grid: nn.ParameterList
        for scale_id,  grid in enumerate(ms_grids[:num_levels]):
            interp_space = 1.
            for ci, coo_comb in enumerate(coo_combs):
                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, feature_dim)
                )
                # compute product over planes
                interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

    def get_scaling(self, hidden_feats=None):
        if hidden_feats is not None:
            delta_scaling = self.delta_scaling_network(hidden_feats)
            scaling = self._scaling + delta_scaling
        else:
            scaling = self._scaling
        scaling = self.scaling_activation(scaling)

        return scaling

    def get_rotation(self, hidden_feats=None):
        rot = self._rotation
        if hidden_feats is not None:
            if self.cfg.use_global_trans:
                
                hidden_feats_global = hidden_feats.mean(0, keepdim=True)
             
                global_rot = self.global_rot_network(hidden_feats_global)
                global_rot = self.global_rot_trans_activation(global_rot) * 2 * math.pi - math.pi
                
                global_rot_matrix = euler_angles_to_rotation_matrix(global_rot.squeeze(0))
                
                # rotation: rotate, WITHOUT this step leading to blury renderings!!
                new_rotation = build_rotation(rot)
                new_rotation = global_rot_matrix.to(new_rotation) @ new_rotation
                rot = extract_rotation_torch(new_rotation)

            delta_rot = self.delta_rot_network(hidden_feats)
            rot = rot + delta_rot
        else:
            rot = self._rotation
        rot = self.rotation_activation(rot)

        return rot

    def get_xyz(self, hidden_feats=None):
        xyz = self._xyz 

        if hidden_feats is not None:
            if self.cfg.use_global_trans:
    
                hidden_feats_global = hidden_feats.mean(0, keepdim=True)
             
                global_trans = self.global_trans_network(hidden_feats_global)
                global_trans = self.global_rot_trans_activation(global_trans) * 2 - 1
                
                global_rot = self.global_rot_network(hidden_feats_global)
                global_rot = self.global_rot_trans_activation(global_rot) * 2 * math.pi - math.pi
            
                global_rot_matrix = euler_angles_to_rotation_matrix(global_rot.squeeze(0))
                
                xyz = (global_rot_matrix.to(xyz) @ xyz.T).T
                
                xyz = xyz + global_trans
                
            delta_xyz = self.delta_xyz_network(hidden_feats)
            
            xyz = xyz + delta_xyz

        else:
            xyz = self._xyz

        return xyz
