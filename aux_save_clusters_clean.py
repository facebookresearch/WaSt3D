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
    kmeans = KMeans(n_clusters=k_clusters, n_init=20, max_iter=100)
    cluster_indices = kmeans.fit_predict(points)
    cluster_centers = kmeans.cluster_centers_
    return cluster_indices, cluster_centers




def clustering(opt):
    NUM_CLUSTERS=30
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_lego_spheres_v5_clusters_{NUM_CLUSTERS}/'
    CKPT_PATH="/home/dimakot55/output_data/gs_my/gs/GT_lego_spheres_v5/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_brickwall_spheres_clusters_{NUM_CLUSTERS}/'
    CKPT_PATH="/home/dimakot55/output_data/gs_my/gs/GT_brickwall/chkpnt30000.pth"

    # Rose bush scene
    NUM_CLUSTERS=30
    CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_rose_bush_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_rose_bush_spheres_clusters_{NUM_CLUSTERS}/'
    # Coast rocks cene
    NUM_CLUSTERS=30
    CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new/GT_coast_rocks_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_coast_rocks_spheres_clusters_{NUM_CLUSTERS}/'
    # Rose bush scene
    NUM_CLUSTERS=30
    CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_rose_bush_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_rose_bush_spheres_clusters_{NUM_CLUSTERS}/'

    # Skull scene
    NUM_CLUSTERS=3
    CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_skull_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_skull_spheres_clusters_{NUM_CLUSTERS}/'

    # Pebbles scene
    NUM_CLUSTERS=30
    CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_pebbles_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_pebbles_spheres_clusters_{NUM_CLUSTERS}/'

    # Grass scene
    NUM_CLUSTERS=2
    CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_grass_spheres/chkpnt30000.pth"
    CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_grass_spheres_clusters_{NUM_CLUSTERS}/'

    # Pebbles scene
    # NUM_CLUSTERS=5
    # CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_pebbles_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_pebbles_spheres_clusters_{NUM_CLUSTERS}/'

    # Bark fragment scene
    # NUM_CLUSTERS=5
    # CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_bark_fragment_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_bark_fragment_spheres_clusters_{NUM_CLUSTERS}/'

    # # Rocks4 scene
    # NUM_CLUSTERS=4
    # CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_rocks4_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_rocks4_spheres_clusters_{NUM_CLUSTERS}/'

    # # Anthurium
    # NUM_CLUSTERS=1
    # CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_anthurium_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_anthurium_spheres_clusters_{NUM_CLUSTERS}/'

    # # Skull scene
    # NUM_CLUSTERS=1
    # CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_skull_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_skull_spheres_clusters_{NUM_CLUSTERS}/'

    # Rose bush scene
    # NUM_CLUSTERS=10
    # CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_rose_bush_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_rose_bush_spheres_clusters_{NUM_CLUSTERS}/'

    # Anthurium
    # NUM_CLUSTERS=5
    # CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_anthurium_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_anthurium_spheres_clusters_{NUM_CLUSTERS}/'


    # # Rocks4 scene
    # NUM_CLUSTERS=10
    # CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_rocks4_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_rocks4_spheres_clusters_{NUM_CLUSTERS}/'

    # # Anthurium
    # NUM_CLUSTERS=4
    # CKPT_PATH="/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/GT_anthurium_spheres/chkpnt7000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_anthurium_spheres_clusters_{NUM_CLUSTERS}/'

    # # Bark fragment scene
    # NUM_CLUSTERS=15
    # CKPT_PATH="/home/dimakot55/output_data/style_scenes_spheres_new//GT_bark_fragment_spheres/chkpnt30000.pth"
    # CLUSTERS_DIR=f'/home/dimakot55/output_data/gs_my/GT_bark_fragment_spheres_clusters_{NUM_CLUSTERS}/'




    os.makedirs(args.output_dir, exist_ok=True)
    gaussians = GaussianModel(3)
    #dummy_parameters = torch.nn.Parameter(torch.zeros([1, 3]))
    gaussians.restore(torch.load(args.ckpt_path)[0], opt)



    print("Start clustering scene...")
    gaussians_cluster_indices, gaussians_cluster_centers = cluster_points(gaussians._xyz.detach().cpu().numpy(), args.num_clusters)

    print("End clustering scene.")
    random_clusters_centers = torch.randn(args.num_clusters, 3).to(device="cuda")

    # extract one cluster of the content scene
    with torch.no_grad():
        gaussians._xyz = gaussians._xyz - torch.tensor(gaussians_cluster_centers[gaussians_cluster_indices]).to(device='cuda')
        for cluster_idx in np.unique(gaussians_cluster_indices):
            idcs_cluster_1 = np.where(gaussians_cluster_indices==cluster_idx)[0]

            # Now leave only elements from this cluster_1
            cluster_dict = {}
            for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
                cluster_dict[attr] = getattr(gaussians, attr)[idcs_cluster_1].cpu().numpy()
            np.savez(args.output_dir+f"/cluster_{cluster_idx}.npz", **cluster_dict)

    return




if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Training script parameters")
    op = OptimizationParams(parser)
    parser.add_argument('--ckpt_path', type=str, default="/home/dimakot55/output_data/style_scenes_spheres_new//GT_pebbles_spheres/chkpnt30000.pth")
    parser.add_argument('--output_dir', type=str, default="/home/dimakot55/output_data/gs_my/GT_grass_spheres_clusters_5")
    parser.add_argument('--num_clusters', type=int, default=5)

    args = parser.parse_args(sys.argv[1:])

    clustering(opt=op.extract(args))
    print("Clustering is completed")
