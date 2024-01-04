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

import random

def load_all_images(image_folder="/home/dimakot55/data/nerf_synthetic/lego/train/"):
    from PIL import Image
    import numpy as np
    # Define the common size for resizing
    common_size = (256, 256)  # Adjust as needed

    # Initialize a list to store the images
    image_list = []

    # Loop through the files in the directory
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            # Load the image, convert to RGB, and resize
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(common_size, Image.Resampling.LANCZOS)


            # Append the image to the list
            image_list.append(np.array(image, dtype=np.float32)/256.)
    return image_list

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
    kmeans = KMeans(n_clusters=k_clusters, n_init=1, max_iter=30)
    cluster_indices = kmeans.fit_predict(points)
    cluster_centers = kmeans.cluster_centers_
    return cluster_indices, cluster_centers


def init_content_gaussian(dataset, opt):
    # Inspite of the name it is actually easier to return the style image gaussians just arranged along the
    # content scene coordinates.

    # Load content and style scenes and initialize the gaussians
    gsns_cnt = GaussianModel(dataset.sh_degree)
    scene_cnt = Scene(dataset, gsns_cnt, shuffle=True)
    gsns_cnt.restore(torch.load("/home/dimakot55/output_data/gs_my/GT_hotdog/chkpnt30000.pth")[0], opt)

    gsns_stl = GaussianModel(dataset.sh_degree)
    scene_stl = Scene(dataset, gsns_stl, shuffle=True)
    gsns_stl.restore(torch.load("/home/dimakot55/output_data/gs_my/GT_lego/chkpnt30000.pth")[0], opt)

    # Initialize content image with patches of style scene
    NUM_CLUSTERS=500

    print("Start clustering content scene...")
    gsns_cnt_cluster_idcs, gsns_cnt_cluster_cntrs = cluster_points(gsns_cnt._xyz.detach().cpu().numpy(), NUM_CLUSTERS)
    print("End clustering content scene.")
    print("Start clustering style scene...")
    gsns_stl_cluster_idcs, gsns_stl_cluster_cntrs = cluster_points(gsns_stl._xyz.detach().cpu().numpy(), NUM_CLUSTERS)
    print("End clustering style scene.")

    with torch.no_grad():
        gsns_stl._xyz = gsns_stl._xyz - \
            torch.tensor(gsns_stl_cluster_cntrs[gsns_stl_cluster_idcs]).to(device='cuda') + \
            torch.tensor(gsns_cnt_cluster_cntrs[gsns_stl_cluster_idcs]).to(device='cuda')
    return gsns_stl, scene_stl, gsns_stl_cluster_idcs

def get_intracluster_stats(gaussians, gaussian_indices, attrbs):
    """
    Compute statistics of attributes of elements within a cluster for all
    clusters in the gaussians specifeied in the gaussian indices.

    Args:
    - gaussians (GaussianModel): A Gaussian model containing the data.
    - gaussian_indices (list): List of indices of the gaussians that belong to the cluster. Has the same length as the
      gaussians._xyz tensor and used to enumerate them.
    - attrbs (dict): Dictionary mapping attribute names to their tensors.

    Returns:
    - dict: Dictionary mapping attribute names to the list of differences within every cluster
      as specified by the gaussian_indices.
    """

    # # Define a loss function
    # def loss_function(pairwise_diff_matrices_pred, pairwise_diff_matrices_target=None):
    #     loss = torch.tensor(0.0, device=device)

    #     for k, matrix in enumerate(pairwise_diff_matrices_pred):
    #         if pairwise_diff_matrices_target is not None:
    #             diff = torch.square(matrix - pairwise_diff_matrices_target[k])  # Compute difference from target matrix (e.g., zeros)
    #         else:
    #             diff = torch.square(matrix)  # Compute difference from target matrix (e.g., zeros)
    #         loss += torch.mean(diff)  # Accumulate the loss

    #     return loss

    def compute_pairwise_differences(tensor, indices):
        """
        Group elements based on indices and compute pairwise differences within each group.

        Args:
        - tensor (torch.Tensor): Input tensor of shape [N, d], where N is the number of elements and d is the dimensionality.
        - indices (torch.Tensor or list): List of indices of shape [N] with elements in the range [1, ..., K].

        Returns:
        - list of torch.Tensor: A list of K square matrices containing pairwise differences within each group.
        """
        device = tensor.device
        K = torch.max(indices).item()  # Calculate the number of clusters

        # Initialize a list to store pairwise differences for each group
        pairwise_diff_matrices = []

        for k in range(1, K + 1):
            # Select elements that belong to cluster k
            group_indices = (indices == k).nonzero().view(-1)
            group_elements = tensor[group_indices]

            # Compute pairwise differences within the group (vectorized version)
            # TODO: fix for inputs with multiple dimensions
            pairwise_diff = torch.cdist(group_elements, group_elements, p=2)  # Compute L2 distances

            pairwise_diff_matrices.append(pairwise_diff)

        return pairwise_diff_matrices

    # aggregate all differences within all clusters for all attributes
    if not isinstance(gaussian_indices, torch.Tensor):
        gaussian_indices_tensor = torch.tensor(gaussian_indices, dtype=torch.int32, device="cuda")
    else:
        gaussian_indices_tensor = gaussian_indices
    dict_attr_to_diffs_list = {}
    for attr in attrbs:
        tensor = getattr(gaussians, attr)
        dict_attr_to_diffs_list[attr] = compute_pairwise_differences(tensor, gaussian_indices_tensor)

    return dict_attr_to_diffs_list




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # load all style images
    style_images=load_all_images(image_folder="/home/dimakot55/data/nerf_synthetic/lego/train/")
    # Instantiate the scene.
    # Initialize gaussians
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians, scene, gaussians_clstr_idcs = init_content_gaussian(dataset=dataset, opt=opt)
    gaussians.training_setup(opt)
    print("gaussians._features_dc.shape:", gaussians._features_dc.shape)
    print("gaussians._features_dc[0]:", gaussians._features_dc[0])
    print("gaussians._features_rest.shape:", gaussians._features_rest.shape)
    print("gaussians._features_rest[0]:", gaussians._features_rest[0])
    # Compute initial cluster statistics. Differences between elements within a cluster for all clusters.
    gaussians_clstr_diffs_dict_GT = get_intracluster_stats(
        gaussians,
        gaussians_clstr_idcs,
        # attrbs=["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation"])
        #attrbs=["_xyz", "_scaling", "_rotation"])
        attrbs=["_xyz"])

    for key in gaussians_clstr_diffs_dict_GT.keys():
        print("{}: has {} elements, with first element of shape {}".format(key, len(gaussians_clstr_diffs_dict_GT[key]),
        gaussians_clstr_diffs_dict_GT[key][0].shape))

    # Now detach values of all the tensors in the dictionary so that they are not updated during training
    for key in gaussians_clstr_diffs_dict_GT.keys():
        gaussians_clstr_diffs_dict_GT[key] = [t.clone().detach() for t in gaussians_clstr_diffs_dict_GT[key]]


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    vgg = VGG().to("cuda").eval()

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # loading and saving block
    # iteration = 137
    # print("\n[ITER {}] Saving Checkpoint and gaussians".format(iteration))
    # torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # scene.save(iteration)
    # return

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # with torch.no_grad():
    #     gaussians._scaling = gaussians._scaling * 1. # scale up the initial gaussians
    gaussians._scaling.requires_grad_(False)
    # gaussians._rotation.requires_grad_(False)
    # gaussians._features_dc.requires_grad_(False)
    # gaussians._features_rest.requires_grad_(False)
    # gaussians._opacity.requires_grad_(True)

    # print("add noise to the gaussians")
    # with torch.no_grad():
    #     for attr in ['_xyz',  '_features_dc', '_features_rest', '_rotation', '_opacity' ]: #'_scaling'
    #         val = getattr(gaussians, attr).detach().clone().to(device="cuda")
    #         mean = torch.mean(val, dim=0)
    #         std = torch.std(val, dim=0)
    #         noise = torch.randn_like(val).to(device="cuda")
    #         setattr(gaussians, attr, val + noise * std * 0.1)

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
        #gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if iteration <2:
            print("gt_image")

            print(gt_image.shape)
            print(gt_image)
            print(torch.min(gt_image), torch.max(gt_image))
        # Compute features(VGG) and style and content losses
        image_feats = get_features(input_tensor=image, model=vgg)
        gt_image_feats = get_features(input_tensor=gt_image, model=vgg)
        gt_image_style = random.choice(style_images)
        gt_image_style = torch.tensor(gt_image_style, dtype=torch.float32, device="cuda").permute(2,0,1)

        if iteration <2:
            print("gt_image_style")
            print(gt_image_style.shape)
            print(gt_image_style)
            print(torch.min(gt_image_style), torch.max(gt_image_style))

        style_image_feats = get_features(input_tensor=gt_image_style, model=vgg)

        layers_style = [0,1]
        layers_content = [2,3]

        l_cont = content_loss([image_feats[i] for i in layers_content], [gt_image_feats[i] for i in layers_content])
        l_tv = tv_loss(image)
        l_style = style_loss([image_feats[i] for i in layers_style], [style_image_feats[i] for i in layers_style])


        # Intracluster losses
        if False:
            gaussians_clstr_diffs_dict_pred = get_intracluster_stats(gaussians, gaussians_clstr_idcs,
                attrbs=gaussians_clstr_diffs_dict_GT.keys())

            l_intracluster_dict = {}
            for attr in gaussians_clstr_diffs_dict_pred.keys():
                l_intracluster = 0.0
                for i in range(len(gaussians_clstr_diffs_dict_pred[attr])):
                    l_intracluster += torch.mean(torch.square(gaussians_clstr_diffs_dict_pred[attr][i] - gaussians_clstr_diffs_dict_GT[attr][i]))
                l_intracluster_dict[attr] = l_intracluster / len(gaussians_clstr_diffs_dict_pred[attr]) * 1e0

            if iteration % 10 == 0:
                print("Intracluster Losses: ", l_intracluster_dict)


        l_cont = l_cont * 1e1 # was 1e1
        l_tv = l_tv * 1e3
        l_style = l_style * 1e-3
        loss = l_cont + l_tv + l_style # + sum(l_intracluster_dict.values())

        loss.backward()

        iter_end.record()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                #progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({"tv": f"{l_tv:.{7}f}", "cont": f"{l_cont:.{7}f}", "stl": f"{l_style:.{7}f}", "loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

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
