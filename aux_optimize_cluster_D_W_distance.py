# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


import ot


from sklearn.cluster import KMeans
def cluster_points(points, k_clusters):
    """
    Cluster a set of points into k_clusters using K-Means.

    Args:
    - points (numpy.ndarray or list): Input points of shape [N, d], where N is the number of points and d is the dimensionality.
    - k_clusters (int): The number of clusters.

    Returns:
    - numpy.ndarray: An array of cluster indices for each point of shape [N,].
    - numpy.ndarray: An array of cluster centers of shape [k_clusters, d].
    """
    kmeans = KMeans(n_clusters=k_clusters, n_init=1, max_iter=20)
    cluster_indices = kmeans.fit_predict(points)
    cluster_centers = kmeans.cluster_centers_
    return cluster_indices, cluster_centers




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # Style scene
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_target = GaussianModel(dataset.sh_degree)
    scene_target = Scene(dataset, gaussians_target, shuffle=True)
    gaussians_target.restore(torch.load("/home/dimakot55/output_data/gs_my/GT_lego_cluster2/chkpnt1.pth")[0], opt)

    # Freeze target scene
    gaussians_target._xyz.requires_grad_(False)
    gaussians_target._features_dc.requires_grad_(False)
    gaussians_target._features_rest.requires_grad_(False)
    gaussians_target._scaling.requires_grad_(False)
    gaussians_target._rotation.requires_grad_(False)
    gaussians_target._opacity.requires_grad_(False)

    # Get target matrix of pairwise distance between all coordinates _xyz
    with torch.no_grad():
        D_xyz_target = torch.cdist(gaussians_target._xyz, gaussians_target._xyz)
        D_rotation_target = torch.cdist(gaussians_target._rotation[:,:-1], gaussians_target._xyz) + \
                       torch.cdist(gaussians_target._rotation[:,1:], gaussians_target._xyz)
        D_scaling_target = torch.cdist(gaussians_target._scaling, gaussians_target._xyz)

        # Create a mask to keep the k nearest elements for each row
        k = 10
        sorted_values, _ = torch.sort(D_xyz_target, dim=1)  # Sort each row along dimension 1
        D_xyz_target_mask = (D_xyz_target <= sorted_values[:, k - 1:k])
        D_xyz_target_mask = D_xyz_target_mask.to(dtype=torch.float32)

        print("D_xyz_target_mask.shape", D_xyz_target_mask.shape)
        print("D_xyz_target_mask", D_xyz_target_mask)


    # Init scene to optimize
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.training_setup(opt)

    # Limit gaussians to the target scene. Also preinitialize size ,
    # color, opacity and scaling to those of the target scene.
    N = len(gaussians_target._xyz)
    gaussians._xyz = gaussians._xyz[:N]
    gaussians._features_dc = gaussians_target._features_dc[:N]
    gaussians._features_rest = gaussians_target._features_rest[:N]
    gaussians._scaling = gaussians_target._scaling[:N]
    gaussians._rotation = gaussians_target._rotation[:N]
    gaussians._opacity = gaussians_target._opacity[:N]
    # Fix all but coordinates
    gaussians._xyz.requires_grad_(True)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._scaling.requires_grad_(True)
    gaussians._rotation.requires_grad_(True)
    gaussians._opacity.requires_grad_(False)
    # also make scaling small to have small points. only first
    # with torch.no_grad():
    #     gaussians._scaling = gaussians._scaling * 2.

    print("gaussians._xyz.shape", gaussians._xyz.shape)
    print("gaussians._features_dc.shape", gaussians._features_dc.shape)
    print("gaussians._features_rest.shape", gaussians._features_rest.shape)
    print("gaussians._scaling.shape", gaussians._scaling.shape)
    print("gaussians._rotation.shape", gaussians._rotation.shape)
    print("gaussians._opacity.shape", gaussians._opacity.shape)


    # Init target shape to which we will align our scene

    # Init scene to optimize
    gaussians_shape = GaussianModel(dataset.sh_degree)
    scene_shape = Scene(dataset, gaussians_shape, shuffle=True)
    gaussians_shape.training_setup(opt)

    # Limit gaussians_shape to the target scene. Also preinitialize size ,
    # color, opacity and scaling to those of the target scene.
    N = len(gaussians_target._xyz)
    gaussians_shape._xyz = gaussians_shape._xyz[:N].detach()
    gaussians_shape._features_dc = gaussians_shape._features_dc[:N].detach()
    gaussians_shape._features_rest = gaussians_shape._features_rest[:N].detach()
    gaussians_shape._scaling = gaussians_shape._scaling[:N].detach()
    gaussians_shape._rotation = gaussians_shape._rotation[:N].detach()
    gaussians_shape._opacity = gaussians_shape._opacity[:N].detach()
    # Fix all but coordinates
    # gaussians_shape._xyz.requires_grad_(False)
    # gaussians_shape._features_dc.requires_grad_(False)
    # gaussians_shape._features_rest.requires_grad_(False)
    # gaussians_shape._scaling.requires_grad_(False)
    # gaussians_shape._rotation.requires_grad_(False)
    # gaussians_shape._opacity.requires_grad_(False)
    # also make scaling small to have small points. only first
    # with torch.no_grad():
    #     gaussians_shape._scaling = gaussians_shape._scaling * 2.

    print("gaussians_shape._xyz.shape", gaussians_shape._xyz.shape)
    print("gaussians_shape._features_dc.shape", gaussians_shape._features_dc.shape)
    print("gaussians_shape._features_rest.shape", gaussians_shape._features_rest.shape)
    print("gaussians_shape._scaling.shape", gaussians_shape._scaling.shape)
    print("gaussians_shape._rotation.shape", gaussians_shape._rotation.shape)
    print("gaussians_shape._opacity.shape", gaussians_shape._opacity.shape)


    # now turn them to a sphere
    with torch.no_grad():
        gaussians_shape._xyz[..., -1] = torch.abs(gaussians_shape._xyz[..., -1])
        gaussians_shape._xyz = gaussians_shape._xyz / torch.linalg.norm(gaussians_shape._xyz, dim=-1, keepdim=True)
        # now rescale to have the size of the target scene
        # gaussians_shape._xyz = gaussians_shape._xyz * torch.mean(torch.linalg.norm(gaussians_target._xyz, dim=-1, keepdim=True))

    gaussians_shape._xyz.requires_grad_(False)
    gaussians_shape._features_dc.requires_grad_(False)
    gaussians_shape._features_rest.requires_grad_(False)
    gaussians_shape._scaling.requires_grad_(False)
    gaussians_shape._rotation.requires_grad_(False)
    gaussians_shape._opacity.requires_grad_(False)



    # save data to separate array. Disable for now
    if True:
        np.savez("/home/dimakot55/workspace/gaussian-splatting/notebooks/clusters.npz",
            # target gaussians
            gaussians_target_xyz = gaussians_target._xyz.detach().cpu().numpy(),
            gaussians_target_features_dc = gaussians_target._features_dc.detach().cpu().numpy(),
            gaussians_target_features_rest = gaussians_target._features_rest.detach().cpu().numpy(),
            gaussians_target_scaling = gaussians_target._scaling.detach().cpu().numpy(),
            gaussians_target_rotation = gaussians_target._rotation.detach().cpu().numpy(),
            gaussians_target_opacity = gaussians_target._opacity.detach().cpu().numpy(),
            # randomly initialized gaussians that we want to optimize
            gaussians_xyz = gaussians._xyz.detach().cpu().numpy(),
            gaussians_features_dc = gaussians._features_dc.detach().cpu().numpy(),
            gaussians_features_rest = gaussians._features_rest.detach().cpu().numpy(),
            gaussians_scaling = gaussians._scaling.detach().cpu().numpy(),
            gaussians_rotation = gaussians._rotation.detach().cpu().numpy(),
            gaussians_opacity = gaussians._opacity.detach().cpu().numpy(),
            # shape gaussians
            gaussians_shape_xyz = gaussians_shape._xyz.detach().cpu().numpy(),
            gaussians_shape_features_dc = gaussians_shape._features_dc.detach().cpu().numpy(),
            gaussians_shape_features_rest = gaussians_shape._features_rest.detach().cpu().numpy(),
            gaussians_shape_scaling = gaussians_shape._scaling.detach().cpu().numpy(),
            gaussians_shape_rotation = gaussians_shape._rotation.detach().cpu().numpy(),
            gaussians_shape_opacity = gaussians_shape._opacity.detach().cpu().numpy(),
        )











    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


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
        except Exception as e:
            network_gui.conn = None


    for iteration in range(first_iter, opt.iterations + 1):
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

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        D_xyz = torch.cdist(gaussians._xyz, gaussians._xyz)
        D_rotation = torch.cdist(gaussians._rotation[:,:-1], gaussians._xyz) + \
                      torch.cdist(gaussians._rotation[:,1:], gaussians._xyz)
        D_scaling = torch.cdist(gaussians._scaling, gaussians._xyz)


        # Sample subset of points for which we compute the EMD distance.
        N = len(gaussians._xyz)
        num_samples = 100
        indices = torch.randperm(N)[:num_samples]
        indices_shape = torch.randperm(N)[:num_samples]
        points_gaussians = gaussians._xyz[indices]

        R_target = torch.mean(torch.linalg.norm(gaussians_target._xyz, dim=-1, keepdim=True)).detach()
        points_gaussians_shape = gaussians_shape._xyz[indices_shape] * R_target
        M = ot.dist(points_gaussians, points_gaussians_shape)
        weights = torch.ones(num_samples).to("cuda") / num_samples
        loss_emd = ot.emd2(weights, weights, M)
        loss_emd = loss_emd * 1e0
        #loss_emd = 0.0

        # loss_D_xyz = torch.mean(torch.square(D_xyz - D_xyz_target))
        # loss_D_rotation = torch.mean(torch.square(D_rotation - D_rotation_target))
        # loss_D_scaling = torch.mean(torch.square(D_scaling - D_scaling_target))

        loss_D_xyz = torch.mean(torch.abs(D_xyz - D_xyz_target) * D_xyz_target_mask)
        loss_D_rotation = torch.mean(torch.abs(D_rotation - D_rotation_target) * D_xyz_target_mask)
        loss_D_scaling = torch.mean(torch.abs(D_scaling - D_scaling_target) * D_xyz_target_mask)

        loss = loss_D_xyz + loss_D_rotation + loss_D_scaling + 1.*loss_emd
        #loss = loss_emd
        loss.backward()
        iter_end.record()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss_D_xyz": f"{loss_D_xyz:.{7}f}",
                                          "loss_D_rotation": f"{loss_D_rotation:.{7}f}",
                                          "loss_D_scaling": f"{loss_D_scaling:.{7}f}",
                                          "loss_emd": f"{loss_emd:.{7}f}",
                                          "loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


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
