# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code to add depth visualization in nerfstudio
# should be placed to nerfstudio/nerfstudio/models/gaussian_splatting.py

import os
import json
import numpy as np
import torch
import math
import torchvision
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from torch.nn import Parameter

from kornia.geometry.depth import depth_to_normals

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.fields.gaussian_splatting_field import GaussianSplattingField
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nerfstudio.utils.gaussian_splatting_sh_utils import eval_sh
from nerfstudio.cameras.gaussian_splatting_camera import Camera as GaussianSplattingCamera
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, focal2fov, fov2focal


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(
        default_factory=lambda: GaussianSplatting
    )

    background_color: str = "black"

    sh_degree: int = 3


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    model_path: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            load_iteration: int = -1,
            orientation_transform: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        super().__init__(config, scene_box, num_train_data)

    def populate_modules(self):
        super().populate_modules()

        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        # load gaussian model
        self.gaussian_model = GaussianSplattingField(sh_degree=self.config.sh_degree)

        self.gaussian_model.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.load_iteration),
                                                  "point_cloud.ply"))

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
        )

        render = render_results["render"]
        depth  = render_results["depth"][:, :, None]
        K = torch.eye(3)[None].cuda()
        K[0, 0, 0] = .5 * viewpoint_camera.image_width / np.tan(.5 * float(viewpoint_camera.FoVx))
        K[0, 1, 1] = .5 * viewpoint_camera.image_height / np.tan(.5 * float(viewpoint_camera.FoVy))
        K[0, 0, 2] = .5 * viewpoint_camera.image_width
        K[0, 1, 2] = .5 * viewpoint_camera.image_height
        normal = (depth_to_normals(render_results["depth"][None, None], K)[0].permute(1, 2, 0) + 1) / 2
        # normal = (normal * 255).type(torch.uint8)
        depth[depth < 2] = 1e10
        depth = 1.0 / depth
        # print(render.shape, depth.shape)
        # print(depth.min(), depth.max())

        rgb = torch.permute(torch.clamp(render, max=1.), (1, 2, 0))
        return {
            "rgb1": rgb,
            "depth": depth,
            "rgb": normal,
        }

    def ns2gs_camera(self, ns_camera):
        c2w = torch.clone(ns_camera.camera_to_worlds)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=ns_camera.camera_to_worlds.device)], dim=0)

        # reorient
        if self.orientation_transform is not None:
            c2w = torch.matmul(self.orientation_transform.to(c2w.device), c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            data_device=ns_camera.camera_to_worlds.device,
        )

    @staticmethod
    def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0


        cam_view_depth = torch.zeros_like(pc.get_xyz[:, :1], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

        sampling_offsets = torch.rand(viewpoint_camera.image_height, viewpoint_camera.image_width, 2, device="cuda") * 0
        try:
            screenspace_points.retain_grad()
            cam_view_depth.retain_grad()
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
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, rendered_depth, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            cam_view_depth = cam_view_depth,
            sampling_offsets = sampling_offsets)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii}
