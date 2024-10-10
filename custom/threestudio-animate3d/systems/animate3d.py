import os
from dataclasses import dataclass, field

from PIL import Image
import numpy as np
from einops import rearrange

import random

import threestudio
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn.utils import clip_grad_norm_

from torch.cuda.amp import autocast, GradScaler

from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from animatediff.utils.util import export_to_gif_mv

import shutil
from .util import (
    cal_connectivity_from_points, 
    cal_arap_error, 
    prepare_arap_from_mesh_vertices,
    sample_matrix_vectorized
)

from pytorch3d.ops import knn_points, knn_gather

import json

# from custom.threestudio_3dgs.geometry.gaussian_base import BasicPointCloud
import sys
BasicPointCloud = getattr(sys.modules["threestudio-3dgs.geometry.gaussian_base"], 'BasicPointCloud')

     
@threestudio.register("gaussian-splatting-animate3d-system")
class Animate3DSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        
        training: bool = True
        load_guidance: bool = True
        guidance_eval_feq: int = 0
        
        n_view: int = 4
        n_frame: int = 8
        progressive_iter_per_frame: int = 150
      
        test_option: str = "testset" # choose from four_view/testset
        save_gaussian_trajectory: bool = False
        
        connected_vertices_info_path: str = "" # for mesh animation
        
        sample_strategy: str = "normal" # choose from normal/light
        
    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = True

        # is_training = self.cfg.get("training", True)
        if self.cfg.training and self.cfg.load_guidance:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

        if self.cfg.training:
            if self.cfg.connected_vertices_info_path != "":
                    
                with open(self.cfg.connected_vertices_info_path, "r") as file_to_read:
                    connected_vertices_info = json.load(file_to_read)
                
                connected_nn_idx, max_nn_idx = prepare_arap_from_mesh_vertices(connected_vertices_info)
                connected_nn_idx = connected_nn_idx.cuda()
                max_nn_idx = max_nn_idx.cuda()
                
                self.connected_nn_idx = connected_nn_idx
                self.max_nn_idx = max_nn_idx
            

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def training_step(self, batch, batch_idx):

        do_reconstruction = True
        if self.cfg.load_guidance:
            do_guidance = True
        else:
            do_guidance = False

        loss_sds = 0.0
        loss = 0.0
    
        batch["do_guidance"] = do_guidance
        batch["do_reconstruction"] = do_reconstruction
            
         # filter batch, train frame-by-frame
        if do_guidance:
            start_index = self.cfg.n_frame-2
        else:
            start_index = min(self.global_step // self.cfg.progressive_iter_per_frame, self.cfg.n_frame-2)
        
        if do_reconstruction:
            
            if self.cfg.sample_strategy == "normal":
                sampled_frame_idx = [i for i in range(1, start_index+2)]
            elif self.cfg.sample_strategy == "light":
                if start_index == 0:
                    sampled_frame_idx = [1]
                elif self.global_step >= (self.cfg.progressive_iter_per_frame * (self.cfg.n_frame - 1)):
                    sampled_frame_idx = [i for i in range(1, self.cfg.n_frame)]
                else:
                    sampled_frame_idx = [random.randint(1, start_index)] + [start_index+1]
            else:
                raise NotImplementedError(f"sample_strategy {self.cfg.sample_strategy} not supported")
            
            sampled_idx = []
            for view_idx in range(self.cfg.n_view):
                for idx in sampled_frame_idx:
                    sampled_idx.append(view_idx*self.cfg.n_frame + idx)
            
                 
            for key, val in batch.items():
                try:
                    if val.shape[0] == (self.cfg.n_frame * self.cfg.n_view):
                        batch[key] = val[sampled_idx]
                except:
                    pass
            
            out = self(batch)
            pred_rgb = out["comp_rgb"]
            
            # reconstruction loss
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]
            
            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + self.renderer.cfg.back_ground_color[0] * (
                1 - gt_mask.float()
            ) # bg grey 

            loss_rgb = F.mse_loss(gt_rgb, pred_rgb)
            loss += self.C(self.cfg.loss["lambda_rgb"]) * loss_rgb

            # mask loss
            loss_mask = F.mse_loss(gt_mask.float(), out["comp_mask"])
            loss += self.C(self.cfg.loss["lambda_mask"]) * loss_mask

        if do_guidance:
            
            batch["random_camera"]["do_guidance"] = do_guidance
            batch["random_camera"]["do_reconstruction"] = do_reconstruction
            out = self(batch["random_camera"])
            
            guidance_inp = out["comp_rgb"]
            viewspace_point_tensor = out["viewspace_points"]
    
            guidance_eval = (self.cfg.guidance_eval_feq > 0) and (self.global_step % self.cfg.guidance_eval_feq == 0)
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch["random_camera"], rgb_as_latents=False, guidance_eval=guidance_eval,
            )
                
            if guidance_eval:
                guidance_eval_out = guidance_out.pop("eval")
                for eval_name, eval_item in guidance_eval_out.items():
                    if eval_name.startswith('video'):
                        file_name = f"guidance_eval/it{self.global_step}-{eval_name}.gif"
                        save_path = self.get_save_path(file_name)
                        export_to_gif_mv(eval_item, save_path)

            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # arap loss from SCGS: https://github.com/yihua7/SC-GS/blob/master/utils/time_utils.py#L770
        if self.C(self.cfg.loss["lambda_arap"]) > 0.0:
            
            nodes_t = [self.geometry._xyz] + out["means3D"][:len(sampled_frame_idx)]
                
            nodes_t = torch.stack(nodes_t, dim=0)
            
            if self.cfg.connected_vertices_info_path != "":
                
                connected_nn_idx = sample_matrix_vectorized(self.connected_nn_idx, self.cfg.loss.arap_K, self.max_nn_idx)
                # connected_nn_idx = connected_nn_idx[:, :self.cfg.arap_K]
                
                Nv = connected_nn_idx.shape[0]
                K = self.cfg.loss.arap_K
                
                ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K).reshape([-1])
                jj = connected_nn_idx.reshape([-1])
                nn = torch.arange(K)[None].cuda().long().expand(Nv, K).reshape([-1])
                mask = jj != -1
                ii, jj, nn = ii[mask], jj[mask], nn[mask]
                
            else:
                hyper_nodes = nodes_t[:1] # [T, M, 3]
                    
                ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, radius=self.cfg.loss.arap_radius, K=self.cfg.loss.arap_K)  # connectivity of control nodes
            
            error = cal_arap_error(nodes_t, ii, jj, nn, K=self.cfg.loss.arap_K, sample_num=self.cfg.loss.arap_sample_num)

            loss_arap = error
            loss += self.C(self.cfg.loss["lambda_arap"]) * loss_arap
            
        if (
            out.__contains__("comp_normal")
            and self.cfg.loss["lambda_normal_tv"] > 0.0
        ):
            loss_normal_tv = self.C(self.cfg.loss["lambda_normal_tv"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
            )
            loss += loss_normal_tv
            

        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = torch.cat(out["means3D"]).norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_sparsity"] > 0.0:
            loss_sparsity = (out["comp_mask"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.cfg.loss["lambda_scales"] > 0.0:
            scales = torch.cat(out["scales"], dim=0)
            scale_sum = torch.sum(scales)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        if out.__contains__("comp_pred_normal"):
            loss_pred_normal = torch.nn.functional.mse_loss(
                out["comp_pred_normal"], out["comp_normal"].detach()
            )
            loss += loss_pred_normal

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
    
        if do_reconstruction:
            self.log(
                "loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "loss_rgb",
                loss_rgb * self.C(self.cfg.loss["lambda_rgb"]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "loss_mask",
                loss_mask * self.C(self.cfg.loss["lambda_mask"]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        
        if do_guidance:
            self.log(
                "loss_sds",
                loss_sds,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if self.C(self.cfg.loss["lambda_arap"]) > 0.0:

            self.log(
                "loss_arap",
                loss_arap * self.C(self.cfg.loss["lambda_arap"]), 
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            
        if self.C(self.cfg.loss["lambda_normal_tv"]) > 0.0:
            
            self.log(
                "loss_normal_tv",
                loss_normal_tv * self.C(self.cfg.loss["lambda_normal_tv"]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            

        total_loss = loss_sds + loss
     
        return_dict = {"loss": total_loss}
 
        return return_dict

    def validation_step(self, batch, batch_idx):
        
        do_reconstruction = True
        if self.cfg.load_guidance:
            do_guidance = True
        else:
            do_guidance = False

        loss_sds = 0.0
        loss = 0.0

        batch["do_guidance"] = do_guidance
        batch["do_reconstruction"] = do_reconstruction
        
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-val/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-val",
            f"it{self.true_global_step}-val",
            "(\d+)\.png",
            save_format="mp4",
            fps=8,
            name="validation",
            step=self.true_global_step,
        )
        
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def test_step(self, batch, batch_idx):
        
        do_reconstruction = True
        if self.cfg.load_guidance:
            do_guidance = True
        else:
            do_guidance = False
        
        batch["do_guidance"] = do_guidance
        batch["do_reconstruction"] = do_reconstruction
        
        out = self(batch)
        rgb = out["comp_rgb"]
        mask = out["comp_mask"]
        rgba_img = torch.cat([rgb, mask], dim=-1)[0]
        rgba_img = rgba_img.detach().cpu().numpy()
        # rgba_img = rgba_img.transpose(1, 2, 0)
        
        out_rgba = Image.fromarray((rgba_img*255).astype(np.uint8))
        batch_index = batch["index"][0].item()
        
        if self.cfg.test_option == "testset":
            elv_index = batch_index // (self.cfg.n_frame*4)
            azi_index = (batch_index // self.cfg.n_frame) % 4
            frame_index = batch_index % self.cfg.n_frame
            output_folder = os.path.join(self._save_dir, "images", f'elv_{elv_index}_azi_{azi_index}')
            
        elif self.cfg.test_option == "four_view":
            frame_index = batch_index
            output_folder = os.path.join(self._save_dir, "images")
        
        else:
            raise NotImeptedError(f"test_option {self.cfg.test_option} not supported")
        
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{frame_index}.png')
        out_rgba.save(output_path)
        
        if self.cfg.save_gaussian_trajectory:
            if batch_index < self.cfg.n_frame:
                mesh_trajectory_folder = os.path.join(self._save_dir, "mesh_trajectory")
                os.makedirs(mesh_trajectory_folder, exist_ok=True)
                output_mesh_trajectory_path = os.path.join(mesh_trajectory_folder, f"{batch_index}.npy")
                means3D = out["means3D"][0].detach().cpu().numpy()
                np.save(output_mesh_trajectory_path, means3D)

    def on_test_epoch_end(self):
        if self.cfg.test_option == "test_set":
            num_elv, num_azi = 3, 4
            for elv_index in range(num_elv):
                for azi_index in range(num_azi):        
                    self.save_img_sequence(
                        f"elv_{elv_index}_azi_{azi_index}",
                        f"elv_{elv_index}_azi_{azi_index}",
                        "(\d+)\.png",
                        save_format="mp4",
                        fps=8,
                        name="test",
                        step=self.true_global_step,
                    )