import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from pytorch3d.ops import sample_farthest_points

from .gaussian_batch_renderer_4d import Gaussian4DBatchRenderer

@threestudio.register("diff-gaussian-rasterizer-advanced-4d")
class DiffGaussian4D(Rasterizer, Gaussian4DBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

        first_frame_trainable: bool = False
        
    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
                
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )


    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        timestamps=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = False

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry

        is_first_frame = False
        hidden_feats = None
        
        if timestamps is not None:
            
            is_first_frame = (timestamps[kwargs["batch_idx"]] == -1).all().item()
            if self.cfg.first_frame_trainable or (not is_first_frame):
                # timestamps should between [-1, 1]
                pts = torch.cat([pc._xyz, torch.ones_like(pc._xyz[...,0:1])* timestamps[kwargs["batch_idx"]]], dim=-1)  
                hidden_feats = pc.interpolate_ms_features(pts, pc.grids)
            
        # Create zero tensor. We will use it to makes pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc._xyz, dtype=pc._xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz(hidden_feats)
    
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        
        if not kwargs["do_guidance"]:
            scales = pc.get_scaling()
        else:
            scales = pc.get_scaling(hidden_feats)
    
        rotations = pc.get_rotation(hidden_feats)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color
                   
        # TODO: test this
        if not kwargs["do_guidance"]:
      
            random_tmp = torch.rand_like(means3D[:, 0:1])
            random_tmp = (random_tmp < 0.1).float()
           
            means3D_input = means3D * random_tmp + means3D.detach().clone() * (1 - random_tmp)
            scales_input = scales * random_tmp + scales.detach().clone() * (1 - random_tmp)
            rotations_input = rotations * random_tmp + rotations.detach().clone() * (1 - random_tmp)
            
        else:
            means3D_input = means3D
            scales_input = scales
            rotations_input = rotations
            
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D_input if kwargs["do_reconstruction"] else means3D_input.detach(),
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales_input,
            rotations=rotations_input,
            cov3D_precomp=cov3D_precomp,
        )
            
    
        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        results = {
            "render": rendered_image.clamp(0, 1),
            "depth": rendered_depth,
            "mask": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "means3D": means3D,
            "scales" : scales,
            "rotations": rotations,
            "opacities": opacity,

        }
             
        return results