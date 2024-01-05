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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
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
    # vgg = VGG().to("cuda").eval()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

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
        Ll1 = l1_loss(image, gt_image)
        # Pixel loss (original)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss_scaling_std = torch.sqrt(torch.mean(torch.std(gaussians._scaling, dim=1, keepdim=True)))
        # loss_scaling_uniformity = torch.sqrt(torch.mean(torch.std(gaussians._scaling, dim=0, keepdim=True))
        # loss_scaling_std = torch.mean(torch.sqrt(1e-6+torch.std(gaussians._scaling, dim=1, keepdim=True)))
        # loss_scaling_uniformity = torch.mean(torch.sqrt(1e-6+torch.std(gaussians._scaling, dim=0, keepdim=True)))
        loss_scaling_std = torch.mean(torch.std(gaussians._scaling, dim=1, keepdim=True))
        loss_scaling_uniformity = torch.mean(torch.std(gaussians._scaling, dim=0, keepdim=True))
        r = 1.3
        eps = 1e-6
        scaling_residue = torch.max(torch.sigmoid(gaussians._scaling), dim=-1)[0] / (torch.min(torch.sigmoid(gaussians._scaling), dim=-1)[0]+eps)
        # print("scaling_residue.shape:", scaling_residue.shape)
        # print("scaling_residue:", scaling_residue)
        # print("gaussians._scaling.shape:", gaussians._scaling.shape)
        # print("gaussians._scaling:", torch.sigmoid(gaussians._scaling))

        r_tensor = torch.FloatTensor([r]).expand_as(scaling_residue).to(device='cuda')
        loss_scaling_anisotropic = torch.mean(torch.max(scaling_residue, r_tensor) - r_tensor)
        if (iteration % 500 == 0):
            print("Loss loss_scaling_std:", loss_scaling_std.cpu() * 1e2)
            print("Loss loss_scaling_uniformity:", loss_scaling_uniformity.cpu() * 1e2)
            print("Loss loss_scaling_anisotropic:", loss_scaling_anisotropic.cpu() * 1e2)
            print("gaussians._xyz.shape:", gaussians._xyz.shape)
        #loss = loss + loss_scaling_std*1e4 + loss_scaling_uniformity*1e3
        # loss4
        #loss = loss + loss_scaling_std*1e2 + loss_scaling_uniformity*1e1
        # loss5
        #loss = loss + loss_scaling_std*1e0 + loss_scaling_uniformity*1e-1
        # loss6
        # loss = loss + loss_scaling_std*0e0 + loss_scaling_uniformity*0e-1
        # loss7
        # loss = loss + loss_scaling_std*1e-1 + loss_scaling_uniformity*1e-2
        # loss8
        # loss = loss + loss_scaling_std*1e-2 + loss_scaling_uniformity*1e-3

        # # loss9
        # loss = loss + loss_scaling_std*1e0 + loss_scaling_uniformity*1e-1
        # loss10
        loss = loss + loss_scaling_anisotropic * 1e-1


        # Compute features(VGG) and style and content losses
        # image_feats = get_features(input_tensor=image, model=vgg)
        # gt_image_feats = get_features(input_tensor=gt_image, model=vgg)
        # l_cont = content_loss(image_feats, gt_image_feats)
        # l_tv = tv_loss(image)
        # l_style = style_loss(image_feats, gt_image_feats)

        # loss = l_cont * 1e-3 + l_tv * 1e0

        # loss = l_cont * 1e-3

        # if (iteration % 30 == 0) and False:
        #     print("Loss TV:", l_tv.cpu() * 1e0)
        #     print("Loss content:", l_cont.cpu() * 1e-3)
        #     print("Loss style:", l_cont.cpu() * 1e-3)

        # loss = l_style * 1e-2

        loss.backward()


        iter_end.record()

        with torch.no_grad():
            # gaussians_mean_scale = torch.mean(gaussians._scaling, dim=0, keepdim=True)
            # gaussians._scaling = gaussians._scaling * 0. + gaussians_mean_scale

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
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
    saving_steps = [1_000, 7_000, 30_000]
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
