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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, VGG, content_loss, style_loss, get_features, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import nerf2nerf.networks
from utils.general_utils import build_scaling_rotation
import numpy as np
from PIL import Image


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import random
import kornia




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # Instantiate the scene.
    # Initialize gaussians
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.restore(torch.load("/home/dimakot55/output_data/gs_my/GT_lego/chkpnt30000.pth")[0], opt)
    #gaussians.training_setup(opt)
    with torch.no_grad():
        gaussians._scaling *= 1.

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)



    viewpoint_stack = None
    ema_loss_for_log = 0.0


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, 10):

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, depth= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        # image_rgb = image[:, :, :3].clone().permute(1, 2, 0).contiguous().cpu().numpy()
        # image_depth = depth[:, :].clone().permute(1, 2, 0).contiguous().cpu().numpy()


        # compute laplacian of shape
        print("start computing normals")
        print("viewpoint_cam:", viewpoint_cam)
        K = torch.eye(3)[None].to(device="cuda")
        K = torch.tensor([[1111, 0, 400],
                          [0, 1111, 400],
                          [0, 0,    1  ]], dtype=torch.float32).unsqueeze(0).to(device="cuda")
        print("K:", K)
        print("depth.shape:", depth.shape)
        normals = kornia.geometry.depth.depth_to_normals(depth=depth.unsqueeze(0).unsqueeze(0),
                                                         camera_matrix=K,
                                                         normalize_points=False)
        print("normals.shape:", normals.shape)


        image_normals = normals.squeeze(0)

        # with torch.no_grad():
        #     image_normals= image_normals#[:, 300:500, 300:500]


        print("image_normals.shape", image_normals.shape)
        if True:
            mins = torch.amin(image_normals, dim=(1, 2), keepdim=True)
            maxs = torch.amax(image_normals, dim=(1, 2), keepdim=True)
            # Normalize the tensor for each color channel
            arr = (image_normals - mins) / (maxs - mins + 1e-6)
            arr = arr.detach().cpu().numpy()

        else:
            arr = (image_normals.detach().cpu().numpy() +1.)/2.


        #arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))*255.

        arr = np.transpose(arr, (1,2,0))*255.
        image_normals = Image.fromarray(arr.astype(np.uint8))
        image_normals.save(f"./extra_results/image_depth_lego_{iteration}_normals_C_normed2_fl=1e5.jpg")


        image_rgb = image.detach().cpu().numpy()
        image_depth = depth.detach().cpu().numpy()

        np.savez(f'./extra_results/image_depth_lego_{iteration}.npz', image=image_rgb, depth=image_depth)
        image_rgb = Image.fromarray((np.clip(np.transpose(image_rgb, (1,2,0)),0.,1.)*255.).astype(np.uint8))
        image_rgb.save(f"./extra_results/image_depth_lego_{iteration}_rgb.jpg")
        #image_depth = Image.fromarray(image_depth, mode="L")
        arr = (image_depth-np.min(image_depth))/(np.max(image_depth)-np.min(image_depth))*255.
        image_depth = Image.fromarray(arr.astype(np.uint8))
        image_depth.save(f"./extra_results/image_depth_lego_{iteration}_depth.jpg")

        # Save central crop
        image_rgb = image.detach().cpu().numpy()
        image_depth = depth.detach().cpu().numpy()

        image_rgb = Image.fromarray((np.clip(np.transpose(image_rgb[:,300:500,300:500], (1,2,0)),0.,1.)*255.).astype(np.uint8))
        image_rgb.save(f"./extra_results/image_depth_lego_{iteration}_rgb_crop.jpg")
        #image_depth = Image.fromarray(image_depth, mode="L")
        image_depth = image_depth[300:500,300:500]
        arr = (image_depth-np.min(image_depth))/(np.max(image_depth)-np.min(image_depth))*255.
        image_depth = Image.fromarray(arr.astype(np.uint8))
        image_depth.save(f"./extra_results/image_depth_lego_{iteration}_depth_crop.jpg")




def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    saving_steps = [1, 100, 300, 1_000, 3_000, 7_000, 10_000, 15_000, 20_000, 30_000]
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=saving_steps)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=saving_steps)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=saving_steps)# default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(dataset=lp.extract(args),
             opt=op.extract(args),
             pipe=pp.extract(args),
             testing_iterations=args.test_iterations,
             saving_iterations=args.save_iterations,
             checkpoint_iterations=args.checkpoint_iterations,
             checkpoint="/home/dimakot55/output_data/gs_my/GT_lego/chkpnt30000.pth",
             debug_from=args.debug_from)
    # rewrite the function above and rename all the input arguments to make it more readable

    # All done
    print("\nTraining complete.")
