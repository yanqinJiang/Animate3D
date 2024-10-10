import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

import sys
BasicPointCloud = getattr(sys.modules["threestudio-3dgs.geometry.gaussian_base"], 'BasicPointCloud')
Camera = getattr(sys.modules["threestudio-3dgs.geometry.gaussian_base"], 'Camera')


class Gaussian4DBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []
        means3Ds = []
        scales = []
        rotations = []
        opacities = []


        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            
            with autocast(enabled=False):
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
                )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                means3Ds.append(render_pkg["means3D"])
                scales.append(render_pkg["scales"])
                rotations.append(render_pkg["rotations"])
                opacities.append(render_pkg["opacities"])

                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
            "means3D": means3Ds,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }

        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
            
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )

        return outputs
