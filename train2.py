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


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Init VGG
    vgg = VGG().to("cuda").eval()
    n2n = nerf2nerf.networks.SphereProjectionModel(input_dim=3, hidden_dim=128, output_dim=3, max_seq_len=16)
    n2n = n2n.to("cuda").eval()


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    print("first_iter, opt.iterations + 1:", first_iter, opt.iterations + 1)
    for iteration in range(10):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        print("iteration:", iteration)

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        print("\n\n\nBefore update")
        print(gaussians._xyz[:10])
        print("gaussians._xyz.shape:", gaussians._xyz.shape)
        print("gaussians._scaling.shape:", gaussians._scaling.shape)
        print("gaussians._rotation.shape:", gaussians._rotation.shape)

        gaussians_S = build_scaling_rotation(s=gaussians._scaling, r=gaussians._rotation)
        print("gaussians_S.shape:", gaussians_S.shape)
        print(gaussians_S[:3])

        #gaussians._xyz = gaussians._xyz * torch.tensor([[0.95, 1.1, 1.2]]).to(device="cuda")


        # fix scaling. Make it the same for all points and for all three axises. In such a way turning gaussians into ellipsoids.
        if False:

            gaussians_mean_scale = torch.mean(gaussians._scaling, dim=0, keepdim=True)
            gaussians._scaling = gaussians._scaling * 0. + gaussians_mean_scale * 0.7

        means_orig = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        std_orig = torch.std(gaussians._xyz, dim=0, keepdim=True)
        #print("means_orig:", means_orig)
        #print("std_orig:", std_orig)
        gaussians._xyz = n2n((gaussians._xyz - means_orig) / std_orig)


        gaussians._xyz = (gaussians._xyz - torch.mean(gaussians._xyz, dim=0, keepdim=True)) / torch.std(gaussians._xyz, dim=0, keepdim=True)

        gaussians._xyz = gaussians._xyz * std_orig + means_orig
        print("\n\n\nafter update")
        print(gaussians._xyz[:10])



        with torch.no_grad():

            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            last_iter_nmbr = 31_000 #first_iter+iteration
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(last_iter_nmbr) + ".pth")
            print("\n[ITER {}] Saving Gaussians".format(last_iter_nmbr))
            scene.save(last_iter_nmbr)
            break

            # Progress bar
            pass

            # Log and save
            pass

            # Densification
            pass

            # Optimizer step
            pass

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                last_iter_nmbr = 31_000 #first_iter+iteration
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(last_iter_nmbr) + ".pth")
                print("\n[ITER {}] Saving Gaussians".format(last_iter_nmbr))
                scene.save(last_iter_nmbr)
                break


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
             checkpoint="/home/dimakot55/output_data/gs_my/GT_lego_spheres/chkpnt30000.pth",
             debug_from=args.debug_from)
    # rewrite the function above and rename all the input arguments to make it more readable

    # All done
    print("\nTraining complete.")
