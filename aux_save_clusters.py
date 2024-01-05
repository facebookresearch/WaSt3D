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
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=True)
    gaussians.restore(torch.load("/home/dimakot55/output_data/gs_my/GT_lego/chkpnt30000.pth")[0], opt)
    gaussians.training_setup(opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    viewpoint_stack = None
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


    NUM_CLUSTERS=100

    print("Start clustering scene...")
    gaussians_cluster_indices, gaussians_cluster_centers = cluster_points(gaussians._xyz.detach().cpu().numpy(), NUM_CLUSTERS)
    print("End clustering scene.")
    random_clusters_centers = torch.randn(NUM_CLUSTERS, 3).to(device="cuda")
    with torch.no_grad():
        # extract one cluster of the content scene
        gaussians._xyz = gaussians._xyz - torch.tensor(gaussians_cluster_centers[gaussians_cluster_indices]).to(device='cuda')
        print("gaussians_cluster_indices:", gaussians_cluster_indices)
        idcs_cluster_1 = np.where(gaussians_cluster_indices==0)[0]
        print("idcs_cluster_1:", idcs_cluster_1)
        # Now leave only elements from this cluster_1
        for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            setattr(gaussians, attr, getattr(gaussians, attr)[idcs_cluster_1])



    with torch.no_grad():

        print("\n[ITER {}] Saving Checkpoint".format(1))
        last_iter_nmbr = 1 #first_iter+iteration
        torch.save((gaussians.capture(), 1), scene.model_path + "/chkpnt" + str(last_iter_nmbr) + ".pth")
        print("\n[ITER {}] Saving Gaussians".format(last_iter_nmbr))
        scene.save(last_iter_nmbr)


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
