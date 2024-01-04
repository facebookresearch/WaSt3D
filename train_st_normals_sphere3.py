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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import kornia
from PIL import Image
import numpy as np
import random

def load_style_normals_images():
    img_list = []
    for iteration in range(1,10):
        img = Image.open(f"./extra_results/image_depth_lego_{iteration}_normals.jpg")
        img_list.append(np.array(img, dtype=np.float32) / 255.0)
    return img_list


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    # Load style images
    style_img_list = load_style_normals_images()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
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
    #vgg = VGG().to("cuda").eval()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    with torch.no_grad():
        # Make larger
        #gaussians._xyz *= 2.

        # Sphere
        # Halfsphere suboption
        #gaussians._xyz[..., -1] = torch.abs(gaussians._xyz[..., -1])
        #gaussians._xyz = gaussians._xyz / torch.linalg.norm(gaussians._xyz, dim=-1, keepdim=True)
        # gaussians._xyz = gaussians._xyz / (torch.linalg.norm(gaussians._xyz, dim=-1, keepdim=True) -
        #  torch.tanh(torch.linalg.norm(gaussians._xyz, dim=-1, keepdim=True))*1e-3)

        # gaussians._xyz = torch.clip(gaussians._xyz, -1.0, 1.0)

        gaussians._xyz *= 1.0
        gaussians._xyz[:, -2] *= 0.01

        #gaussians._xyz[:, -1] *= 0.
        pass
        #gaussians._xyz = gaussians._xyz / (0.5 + torch.linalg.norm(gaussians._xyz, dim=-1, keepdim=True))
        #gaussians._scaling = gaussians._scaling * 1.
        gaussians._opacity = gaussians._opacity*1.# + 1.
        gaussians._scaling = gaussians._scaling*1.0


    # gaussians._xyz.requires_grad_(True)
    # gaussians._features_dc.requires_grad_(False)
    # gaussians._features_rest.requires_grad_(False)
    # gaussians._scaling.requires_grad_(False)
    # gaussians._rotation.requires_grad_(False)
    # gaussians._opacity.requires_grad_(False)



    # for param_group in gaussians.optimizer.param_groups:
    #     param_group['lr'] = 1e6

    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        #print("viewpoint_stack:", viewpoint_stack)
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


    # GT_image_rgb = Image.open(f"./extra_results/image_depth_lego_{1}_rgb.jpg")
    GT_image_rgb = Image.open(f"./extra_data/diamond_pattern.jpg").convert("RGB").resize((800,800))
    GT_image_depth = Image.open(f"./extra_results/image_depth_lego_{1}_depth.jpg")
    GT_image_depth = Image.open(f"./extra_data/diamond_pattern.jpg").resize((800,800))
    GT_image_rgb = np.array(GT_image_rgb, dtype=np.float32) / 255. / 2. + 0.25
    GT_image_depth = np.array(GT_image_depth, dtype=np.float32) / 255. / 2. + 0.25

    # For lego_color pattern
    GT_image_rgb = Image.open(f"./extra_data/lego_color.jpg").convert("RGB").resize((800,800))
    GT_image_depth = Image.open(f"./extra_data/lego_color.jpg").resize((800,800))
    GT_image_rgb = np.array(GT_image_rgb, dtype=np.float32) / 255. / 2. + 0.25
    GT_image_depth = np.array(GT_image_depth, dtype=np.float32) / 255. / 2. + 0.25
    GT_image_depth = np.mean(GT_image_depth, axis=-1)




    image_GT = torch.tensor(GT_image_rgb, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    depth_GT = torch.tensor(GT_image_depth, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image_depth_init = render_pkg["depth"]

        image_depth_init = torch.clone(image_depth_init).detach()

        print(torch.min(image_depth_init), torch.max(image_depth_init), torch.max(image_depth_init))


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

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()



        # SAMPLING OF CAMERA ANGLE AND IMAGE WAS HERE



        #print("img_style.shape:", img_style.shape)
        #viewpoint_cam = viewpoint_stack.pop()
        #print("viewpoint_cam.R, viewpoint_cam.T", viewpoint_cam.R, viewpoint_cam.T)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        R_orig = np.array(viewpoint_cam.R)
        T_orig = np.array(viewpoint_cam.T)
        viewpoint_cam.R += np.random.randn(*viewpoint_cam.R.shape) * 1.1
        viewpoint_cam.T += np.random.randn(*viewpoint_cam.T.shape) * 1.1

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image_depth = render_pkg["depth"]



        def invert_val(val):
            return 1./(val+1.)
        l_rgb = torch.mean((image - image_GT) ** 2)
        # l_depth = torch.mean((image_depth - depth_GT) ** 2)

        #depth_GT_simple =  torch.mean(image_depth) + (depth_GT - 0.5) / 2.
        #depth_GT_simple =  image_depth_init + (depth_GT - 0.5) / 5.
        depth_GT_simple =  image_depth_init * (1. + (depth_GT - torch.mean(depth_GT)) * .5)


        l_depth = torch.mean((image_depth - depth_GT_simple) ** 2)# * (image_depth_init>0.5))
        l_depth = torch.mean((image_depth - depth_GT_simple) ** 2)
        l_depth = torch.mean((image_depth-1.) ** 2)
        l_depth = torch.mean((image_depth-depth_GT) ** 2)



        #l_depth = torch.mean((invert_val(image_depth) - invert_val(depth_GT)) ** 2)
        # l_depth = torch.mean(image_depth)
        l_tv = tv_loss(image)
        l_tv_depth = tv_loss(image_depth)

        l_rgb = l_rgb * 1. #1e0
        l_depth = l_depth * 1e1#  1e0
        l_tv = l_tv * 1e0 #1e0
        l_tv_depth = l_tv_depth * 1e-1#  1e3
        loss = l_depth + l_tv_depth + l_rgb + l_tv

        # loss = l_tv



        loss.backward()
        iter_end.record()

        viewpoint_cam.R = R_orig
        viewpoint_cam.T = T_orig


        if iteration % 1000 == 0:
            # Predicition
            image_rgb = image.detach().cpu().numpy()
            image_depth = image_depth.detach().cpu().numpy()

            np.savez(f'./extra_results2/image_depth_lego_{iteration}_pred.npz', image=image_rgb, depth=image_depth)
            image_rgb = Image.fromarray((np.clip(np.transpose(image_rgb, (1,2,0)),0.,1.)*255.).astype(np.uint8))
            image_rgb.save(f"./extra_results2/image_depth_lego_{iteration}_rgb_pred.jpg")
            #image_depth = Image.fromarray(image_depth, mode="L")
            arr = (image_depth-np.min(image_depth))/(np.max(image_depth)-np.min(image_depth))*255.
            image_depth = Image.fromarray(arr.astype(np.uint8))
            image_depth.save(f"./extra_results2/image_depth_lego_{iteration}_depth_pred.jpg")



            # GT
            image_rgb = image_GT.detach().cpu().numpy()
            image_depth = depth_GT.detach().cpu().numpy()

            np.savez(f'./extra_results2/image_depth_lego_{iteration}_GT.npz', image=image_rgb, depth=image_depth)
            image_rgb = Image.fromarray((np.clip(np.transpose(image_rgb, (1,2,0)),0.,1.)*255.).astype(np.uint8))
            image_rgb.save(f"./extra_results2/image_depth_lego_{iteration}_rgb_GT.jpg")
            #image_depth = Image.fromarray(image_depth, mode="L")
            arr = (image_depth-np.min(image_depth))/(np.max(image_depth)-np.min(image_depth))*255.
            image_depth = Image.fromarray(arr.astype(np.uint8))
            image_depth.save(f"./extra_results2/image_depth_lego_{iteration}_depth_GT.jpg")




        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"tv": f"{l_tv:.{7}f}",
                                          "tv_depth": f"{l_tv_depth:.{7}f}",
                                          "rgb": f"{l_rgb:.{7}f}",
                                          "depth": f"{l_depth:.{7}f}",
                                          "loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    saving_steps = [100, 300, 1_000, 3_000, 7_000, 10_000, 15_000, 20_000, 30_000]
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
