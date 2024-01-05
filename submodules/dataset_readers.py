# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use this code to read LLFF dataset.

import os
import sys
from PIL import Image
from typing import NamedTuple
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import json
import re
from pathlib import Path
from plyfile import PlyData, PlyElement
from kornia.geometry.depth import depth_to_3d, depth_to_normals
import torch
import yaml

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from bilateral_filtering import sparse_bilateral_filtering

from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    render_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"] :
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, c2w=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    radii2 = vertices['radius2'][..., None]
    # ray_dirs = np.vstack([vertices['rx'], vertices['ry'], vertices['rz']]).T
    cams = vertices['cam']
    return BasicPointCloud(points=positions, colors=colors, normals=normals, radii2=radii2, cams=cams)

def storePly(path, xyz, rgb, normals, radii2, cam_idx):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('radius2', 'f4'),
            # ('rx', 'f4'), ('ry', 'f4'), ('rz', 'f4'),
            # Using unsigned int 1. Range 0-255 for camera indices
            ('cam', 'u1')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, radii2, cam_idx), axis=1)
    print(xyz.shape, normals.shape, rgb.shape, radii2.shape, cam_idx.shape)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    print(cam_infos)
    exit()

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


config = yaml.load(open('argument.yaml', 'r'))

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", train=False):
    cam_infos = []
    xyz_arr = []
    rgb_arr = []
    radii2_arr = []
    normals_arr = []
    c2w_arr = []
    ray_dir_arr = []
    cam_idx_arr = []
    indices_arr = []
    depth_arr = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Image processing partial

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            if train:
                ##### Depth processing #####
                # depth = cv2.imread(f'{path}/{frame["file_path"]}_depth_0029.exr',  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

                # disparity, _ = read_pfm(f'{path}/{frame["file_path"]}-dpt_beit_large_512.pfm')
                # depth = 3 * (1 - disparity / 65535)

                depth = np.array(Image.open(f'{path}/{frame["file_path"]}.png_output.png')) / 256
                depth = 3 * depth
                # depth = np.load(f'{path}/{frame["file_path"]}_magnet.npy')
                # depth = 4 * depth
                # print(depth.max(), depth.min())
                disparity = 1/depth
                vis_photos, vis_depths = sparse_bilateral_filtering(disparity.copy(), im_data.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
                disparity = vis_depths[-1]
                depth = 1/disparity
                height, width = depth.shape
                depth = torch.Tensor(depth)[None, None] #/ 1.1

                normal = cv2.imread(f'{path}/{frame["file_path"]}_normal_0029.exr',  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                normal = normal / np.linalg.norm(normal, axis=-1)[..., None]

                focal = .5 * width / np.tan(.5 * float(fovx))

                # Init radius equal to shorter length of the rectangle. Default: Height
                fovy = focal2fov(fov2focal(fovx, width), height)

                # Radii per frame
                radii = np.tan(0.5 * float(fovy))  * depth / height
                # radii2 = radii * 1e-3
                radii2 = radii**2

                K = torch.eye(3)[None]
                K[:, 0, 0] = focal
                K[:, 0, 2] = width / 2.0
                K[:, 1, 1] = focal
                K[:, 1, 2] = height / 2.0

                camera3d = depth_to_3d(depth, K)
                # print(camera3d[:, :2]/camera3d[:, 2:], camera3d.shape)

                xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3).numpy()
                # abc = 0#160000
                # print(xyz_cam[abc:abc+10] / xyz_cam[abc:abc+10, -1:])
                # exit()
                rgb = torch.Tensor(im_data).reshape(-1, 4).numpy()
                normal = torch.Tensor(normal).reshape(-1, 3).numpy()

                radii2 = radii2[0].permute(1, 2, 0).reshape(-1).numpy()

                dep = depth[0].permute(1, 2, 0).reshape(-1).numpy()
                indices = np.argwhere(rgb[:, 3] == 255)[:, 0]
                xyz_cam_valid = xyz_cam[indices]
                rgb_valid = rgb[indices][..., :3]
                radii2_valid = radii2[indices][..., None]
                normal_valid = normal[indices]
                normal_valid_w = normal_valid# @ w2c

                # ray_dir = np.linalg.norm(xyz_cam_valid, axis=1)

                # xyz_homo = np.ones((xyz_cam_valid.shape[0], 4))
                # xyz_homo[:, :3] = xyz_cam_valid
                # xyz_world_homo = np.matmul(c2w, xyz_homo.T).T
                # xyz_world = xyz_world_homo[:, :3] / xyz_world_homo[:, 3:]
                # xyz_arr.append(xyz_world)#[::40])
                xyz_arr.append(xyz_cam_valid)#[::40])
                rgb_arr.append(rgb_valid)#[::40])
                radii2_arr.append(radii2_valid)#[::40])
                normals_arr.append(normal_valid_w)#[::40])
                # c2w_arr.append(np.repeat(c2w[None], xyz_world.shape[0], axis=0))
                # ray_dir_arr.append(ray_dir)
                cam_idx_arr.append(np.repeat(idx, xyz_cam_valid.shape[0]))
                indices_arr.append(torch.tensor(indices + idx*width*height).cuda())
                depth_arr.append(depth)
                ############################

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            # print(image.size)
            # if train:
            #     # anti aliasing trick
            #     factor = 2
            #     image = image.resize((factor * image.size[0], factor * image.size[1]))
            #     cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            #                 image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            # else:
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    if train:
        return cam_infos, xyz_arr, rgb_arr, radii2_arr, normals_arr, cam_idx_arr, indices_arr, depth_arr
    else:
        return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos, xyz_arr, rgb_arr, radii2_arr, normals_arr, cam_idx_arr, indices_arr, depth_arr = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, train=True)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, "point_cloud.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)

    inddd = 0
    # xyz = xyz_arr[inddd]#np.concatenate(xyz_arr, axis=0)#[100:103]
    # rgb = rgb_arr[inddd]#np.concatenate(rgb_arr, axis=0)#[100:103]
    # normals = normals_arr[inddd]#np.concatenate(normals_arr, axis=0)#[100:
    # radii2 = radii2_arr[inddd]#np.concatenate(radii_arr, axis=0)#[100:103]
    # cam_idx = cam_idx_arr[inddd]#np.concatenate(cam_idx_arr, axis
    # indices_arr = indices_arr[inddd]
    # # c2w = c2w_arr[0]#np.concatenate(w2c_arr, axis=0)#[1
    # # ray_dir = ray_dir_arr[0]#np.concatenate(ray_dir_arr, axis=0)#[1
    xyz = np.concatenate(xyz_arr, axis=0)#[100:103]
    rgb = np.concatenate(rgb_arr, axis=0)#[100:103]
    normals = np.concatenate(normals_arr, axis=0)#[100:
    radii2 = np.concatenate(radii2_arr, axis=0)#[100:103]
    cam_idx = np.concatenate(cam_idx_arr, axis=0)#[100:103]
    indices_arr = torch.cat(indices_arr, dim=0)
    depth = torch.cat(depth_arr, dim=1)

    num_pts = xyz.shape[0]
    print(f"Generating point cloud from depth with ({num_pts}) pts...")

    # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    # shs = np.random.random((num_pts, 3)) / 255.0
    # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    storePly(ply_path, xyz, rgb, normals, radii2, cam_idx[..., None])
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, indices_arr, depth

def readDTUInfo(path, white_background, eval, extension=".png"):
    def rescale_poses(poses):
        """Rescales camera poses according to maximum x/y/z value."""
        s = np.max(np.abs(poses[:, :3, -1]))
        out = np.copy(poses)
        out[:, :3, -1] /= s
        return out


    def recenter_poses(poses):
        """Recenter poses around the origin."""
        cam2world = poses_avg(poses)
        poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
        return unpad_poses(poses)

    def poses_avg(poses):
        """New pose using average position, z-axis, and up vector of input poses."""
        position = poses[:, :3, 3].mean(0)
        z_axis = poses[:, :3, 2].mean(0)
        up = poses[:, :3, 1].mean(0)
        cam2world = viewmatrix(z_axis, up, position)
        return cam2world


    def viewmatrix(lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def pad_poses(p):
        """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
        bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
        return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


    def unpad_poses(p):
        """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
        return p[Ellipsis, :3, :4]


    def shift_origins(origins, directions, near=0.0):
        """Shift ray origins to near plane, such that oz = near."""
        t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
        origins = origins + t[Ellipsis, None] * directions
        return origins

    def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
        """Area downsample img (factor must evenly divide img height and width)."""
        sh = img.shape
        max_fn = lambda x: max(x, patch_size)
        out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
        img = cv2.resize(img, out_shape, mode)
        return img



    def focus_pt_fn(poses):
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt


    def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
        """Calculates a forward facing spiral path for rendering for DTU."""

        # Get radii for spiral path using 60th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), perc, 0)
        radii = np.concatenate([radii, [1.]])

        # Generate poses for spiral path.
        render_poses = []
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.]
            position = cam2world @ t
            render_poses.append(viewmatrix(z_axis, up, position, True))
        render_poses = np.stack(render_poses, axis=0)
        return render_poses

    n_images = 49
    dtu_light_cond = 3
    factor = 4
    n_input_views = 3
    white_background = False

    cam_infos = []

    # Loop over all images.
    for i in tqdm(range(1, n_images + 1)):
        # Set light condition string accordingly.
        light_str = f'{dtu_light_cond}_r' + ('5000' if i < 50 else
                                                    '7000')

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        # Load image.
        im_path = os.path.join(path, f'rect_{i:03d}_{light_str}.png')
        with open(im_path, 'rb') as imgin:
            im_data = np.array(Image.open(imgin).convert('RGB'))
            norm_data = im_data / 255.0
            # if factor > 0:
            #     norm_data = downsample(norm_data, factor, patch_size=8)
            image = Image.fromarray(np.array(norm_data*255.0, dtype=np.byte), "RGB")

        width, height = image.size
        # print(image.size)
        cam_info = CameraInfo(uid=i, R=None, T=None, FovY=None, FovX=None, image=image,
                                image_path=im_path, image_name=f'rect_{i:03d}_{light_str}.png', width=width, height=height)

        cam_infos.append(cam_info)

    # print(len(cam_infos))
    camtoworlds = []
    for i in range(1, n_images + 1):
        # Load projection matrix from file.
        fname = f'{path}/../../Calibration/cal18/pos_{i:03d}.txt'
        with open(fname, 'rb') as f:
            projection = np.loadtxt(f, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        # print(camera_mat)

        camera_mat = camera_mat / camera_mat[2, 2]
        # print("camera_mat: ", camera_mat)
        if factor > 0:
            # Scale camera matrix according to downsampling factor.
            camera_mat = np.diag(
                [1./factor,
                1./factor, 1.]).astype(np.float32) @ camera_mat
        focal_length_x = camera_mat[0, 0]
        focal_length_y = camera_mat[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        cam_infos[i-1] = cam_infos[i-1]._replace(FovX=FovX)
        cam_infos[i-1] = cam_infos[i-1]._replace(FovY=FovY)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)


    # fix_rotationO = np.array([
    #     [0, -1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    # ],
    #                         dtype=np.float32)
    camtoworlds = np.stack(camtoworlds)
    # Center and scale poses.
    camtoworlds = recenter_poses(camtoworlds)
    camtoworlds = rescale_poses(camtoworlds)
    print(camtoworlds[0])
    print(camtoworlds.shape)
    print(width, height)

    for i in range(1, n_images + 1):
        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = camtoworlds[i-1]
        # pose[:3, 1:3] *= -1
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_infos[i-1] = cam_infos[i-1]._replace(R=R)
        cam_infos[i-1] = cam_infos[i-1]._replace(T=T)

    test_indices = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47]
    train_indices = [22, 25, 28]
    train_indices = train_indices[:n_input_views]
    train_cam_infos = [cam_infos[i] for i in train_indices]
    test_cam_infos  = [cam_infos[i] for i in test_indices]

    render_cam_infos = []
    render_poses = generate_spiral_path_dtu(camtoworlds, n_frames=240)

    for i, ren_pose in enumerate(render_poses):
        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = ren_pose
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=width, height=height)
        render_cam_infos.append(cam_info)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # SETUP train depth etc
    xyz_arr = []
    rgb_arr = []
    radii2_arr = []
    cam_idx_arr = []
    indices_arr = []
    depth_arr = []
    dep_mask_arr = []

    for idx, cam_info in enumerate(train_cam_infos):
        depth = np.array(Image.open(f'{os.path.dirname(cam_info.image_path)}/depth/{cam_info.image_name}_output.png')) / 256
        # if factor > 0:
        #     depth = downsample(depth, factor, patch_size=8)
        # depth = 3 * depth
        # print(depth.shape, depth.max(), depth.min())
        # exit()
        im_data = np.array(cam_info.image, dtype=np.float32) #/ 255.0

        disparity = 1/depth
        vis_photos, vis_depths = sparse_bilateral_filtering(disparity.copy(), im_data.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
        disparity = vis_depths[-1]
        depth_filtered = 1/disparity


        depth_diff = np.abs(depth_filtered - depth)
        trp_indices = np.argwhere(depth_diff.reshape(-1) > 0.01)[:, 0]

        # trp_indices = np.array([0])

        depth = depth_filtered


        dep_masks = torch.tensor(np.load(f'{os.path.dirname(cam_info.image_path)}/masks/{cam_info.image_name.split(".")[0]}.npy')).permute(2, 0, 1)[None]

        depth = torch.Tensor(depth)[None, None]

        # Init radius equal to shorter length of the rectangle. Default: Height
        fovy = focal2fov(fov2focal(cam_info.FovX, width), height)

        # Radii per frame
        radii = np.tan(0.5 * float(fovy))  * depth / height
        radii2 = radii**2

        K = torch.eye(3)[None]
        K[:, 0, 0] = fov2focal(cam_info.FovX, width)
        K[:, 0, 2] = width / 2.0
        K[:, 1, 1] = fov2focal(cam_info.FovY, height)
        K[:, 1, 2] = height / 2.0
        # print(depth.max(), depth.min(), K)
        # exit()

        camera3d = depth_to_3d(depth, K)

        print(depth.shape, im_data.shape, radii2.shape, camera3d.shape, K)
        # exit()

        xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3).numpy()
        rgb = torch.Tensor(im_data).reshape(-1, 3).numpy()

        radii2 = radii2[0].permute(1, 2, 0).reshape(-1).numpy()

        indices = np.arange(width * height)
        xyz_cam_valid = xyz_cam[indices]
        rgb_valid = rgb[indices][..., :3]
        radii2_valid = radii2[indices][..., None]

        xyz_arr.append(xyz_cam_valid)
        rgb_arr.append(rgb_valid)
        radii2_arr.append(radii2_valid)
        cam_idx_arr.append(np.repeat(idx, xyz_cam_valid.shape[0]))
        # indices_arr.append(torch.tensor(indices + idx*width*height).cuda())
        indices_arr.append(torch.tensor(trp_indices + idx*width*height).cuda())
        depth_arr.append(depth)
        dep_mask_arr.append(dep_masks)

    dep_mask_arr = torch.cat(dep_mask_arr, dim=0).float().cuda()
    print(dep_mask_arr.max(), dep_mask_arr.min())

    image_names = [cam_info.image_name.split('.')[0] for cam_info in train_cam_infos]
    print(image_names)
    flows, masks = {}, {}
    for idx, im1 in enumerate(image_names):
        for idx2, im2 in enumerate(image_names):
            if idx == idx2:
                continue

            flows[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}.npy")).cuda()
            masks[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}w.npy")).cuda()

    # inddd = 2
    # xyz = xyz_arr[inddd]
    # rgb = rgb_arr[inddd]
    # radii2 = radii2_arr[inddd]
    # cam_idx = cam_idx_arr[inddd]
    # indices_arr = indices_arr[inddd]
    # depth = depth_arr[inddd]

    xyz = np.concatenate(xyz_arr, axis=0)
    rgb = np.concatenate(rgb_arr, axis=0)
    radii2 = np.concatenate(radii2_arr, axis=0)
    cam_idx = np.concatenate(cam_idx_arr, axis=0)
    indices_arr = torch.cat(indices_arr, dim=0)
    depth = torch.cat(depth_arr, dim=1)


    ply_path = os.path.join(path, "points3d.ply")

    if os.path.exists(ply_path):
        os.remove(ply_path)


    storePly(ply_path, xyz, rgb, np.zeros_like(xyz), radii2, cam_idx[..., None])
    # ply_path = os.path.join(path, "point_cloud.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")

    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 0.5 + np.array([-0.25, -0.25, 1])
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, indices_arr, depth, flows, masks, dep_mask_arr


def readLLFFInfo(path, white_background, eval, extension=".png"):
    def rescale_poses(poses):
        """Rescales camera poses according to maximum x/y/z value."""
        s = np.max(np.abs(poses[:, :3, -1]))
        out = np.copy(poses)
        out[:, :3, -1] /= s
        return out

    def recenter_poses(poses):
        """Recenter poses around the origin."""
        cam2world = poses_avg(poses)
        poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
        return unpad_poses(poses)

    def poses_avg(poses):
        """New pose using average position, z-axis, and up vector of input poses."""
        position = poses[:, :3, 3].mean(0)
        z_axis = poses[:, :3, 2].mean(0)
        up = poses[:, :3, 1].mean(0)
        cam2world = viewmatrix(z_axis, up, position)
        return cam2world

    def viewmatrix(lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def pad_poses(p):
        """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
        bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
        return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)

    def unpad_poses(p):
        """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
        return p[Ellipsis, :3, :4]

    def shift_origins(origins, directions, near=0.0):
        """Shift ray origins to near plane, such that oz = near."""
        t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
        origins = origins + t[Ellipsis, None] * directions
        return origins

    def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
        """Area downsample img (factor must evenly divide img height and width)."""
        sh = img.shape
        max_fn = lambda x: max(x, patch_size)
        out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
        img = cv2.resize(img, out_shape, mode)
        return img

    def focus_pt_fn(poses):
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
        """Calculates a forward facing spiral path for rendering for DTU."""

        # Get radii for spiral path using 60th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), perc, 0)
        radii = np.concatenate([radii, [1.]])

        # Generate poses for spiral path.
        render_poses = []
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
            position = cam2world @ t
            render_poses.append(viewmatrix(z_axis, up, position, True))
        render_poses = np.stack(render_poses, axis=0)
        return render_poses


    # def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
    #     """Calculates a forward facing spiral path for rendering."""
    #     # Find a reasonable 'focus depth' for this dataset as a weighted average
    #     # of near and far bounds in disparity space.
    #     close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    #     dt = .75
    #     focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    #     # Get radii for spiral path using 90th percentile of camera positions.
    #     positions = poses[:, :3, 3]
    #     radii = np.percentile(np.abs(positions), 90, 0)
    #     radii = np.concatenate([radii, [1.]])

    #     # Generate poses for spiral path.
    #     render_poses = []
    #     cam2world = poses_avg(poses)
    #     up = poses[:, :3, 1].mean(0)
    #     for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    #         t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    #         position = cam2world @ t
    #         lookat = cam2world @ [0, 0, -focal, 1.]
    #         z_axis = position - lookat
    #         render_poses.append(viewmatrix(z_axis, up, position))
    #     render_poses = np.stack(render_poses, axis=0)
    #     return render_poses
    def generate_spiral_path(poses, bounds, fix_rot, n_frames=120, n_rots=1, zrate=.5):
        """Calculates a forward facing spiral path for rendering."""
        # Find a reasonable 'focus depth' for this dataset as a weighted average
        # of near and far bounds in disparity space.
        close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
        dt = .75
        focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

        # Get radii for spiral path using 90th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), 90, 0)
        radii = np.concatenate([radii, [1.]])

        print(focal, radii)

        # Generate poses for spiral path.
        render_poses = []
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.]
            position = cam2world @ t
            lookat = cam2world @ [0, 0, focal, 1.]
            z_axis = -position + lookat
            render_poses.append(viewmatrix(z_axis, up, position))
        render_poses = np.stack(render_poses, axis=0)
        return render_poses

    factor = 8
    llffhold = 8
    n_input_views = 3

    cam_infos = []

    # Load images.
    imgdir_suffix = ''
    if factor > 0:
      imgdir_suffix = f'_{factor}'
    else:
      factor = 1
    imgdir = os.path.join(path, 'images' + imgdir_suffix)
    if not os.path.isdir(imgdir):
      raise ValueError(f'Image folder {imgdir} does not exist.')
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]

    for i, imgfile in enumerate(imgfiles):
      with open(imgfile, 'rb') as imgin:
        image = Image.open(imgin).convert('RGB')
        # norm_data = im_data / 255.0
        # image = Image.fromarray(np.array(norm_data*255.0, dtype=np.byte), "RGB")
        image_name = os.path.basename(imgfile)#.split(".")[0]

        width, height = image.size
        # print(image.size)
        cam_info = CameraInfo(uid=i, R=None, T=None, FovY=None, FovX=None, image=image,
                                image_path=imgfile, image_name=image_name, width=width, height=height)

        cam_infos.append(cam_info)


    # Load poses and bounds.
    with open(os.path.join(path, 'poses_bounds.npy'),
                         'rb') as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]

    # Pull out focal length before processing poses.
    focal = poses[0, -1, -1] / factor

    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)

    poses = poses[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    print(f"Bounds of the scene {bounds[0]}")
    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    print(f"Bounds of the scene {bounds[0]}")

    # Center and scale poses.
    camtoworlds = poses
    camtoworlds = recenter_poses(camtoworlds)
    # camtoworlds = rescale_poses(camtoworlds)

    # camtoworldsO = recenter_poses(posesO)

    # FOV processing
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    for i in range(1, camtoworlds.shape[0] + 1):

        cam_infos[i-1] = cam_infos[i-1]._replace(FovX=FovX)
        cam_infos[i-1] = cam_infos[i-1]._replace(FovY=FovY)

        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = camtoworlds[i-1]
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_infos[i-1] = cam_infos[i-1]._replace(R=R)
        cam_infos[i-1] = cam_infos[i-1]._replace(T=T)

    # Select the split.
    all_indices = np.arange(len(cam_infos))
    split_indices = {
        'test': all_indices[all_indices % llffhold == 0],
        'train': all_indices[all_indices % llffhold != 0],
    }

    train_indices = np.linspace(0, split_indices['train'].shape[0] - 1, n_input_views)
    train_indices = [round(i) for i in train_indices]

    train_indices = [split_indices['train'][i] for i in train_indices]
    train_cam_infos = [cam_infos[i] for i in train_indices]
    test_cam_infos  = [cam_infos[i] for i in split_indices['test']]

    # print(train_cam_infos)
    # exit()

    render_cam_infos = []
    render_poses = generate_spiral_path(camtoworlds, bounds, fix_rotation, n_frames=90)# @ fix_rotation
    # render_poses[:, :3, 3] /= scale
    # print(render_poses[0])
    # print(camtoworlds[0], bounds[0])
    # exit()
    # render_poses = render_poses @ fix_rotation

    for i, render_pose in enumerate(render_poses):
        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = render_pose
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=width, height=height)
        render_cam_infos.append(cam_info)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # SETUP train depth etc
    xyz_arr = []
    rgb_arr = []
    radii2_arr = []
    cam_idx_arr = []
    indices_arr = []
    depth_arr = []
    dep_mask_arr = []

    for idx, cam_info in enumerate(train_cam_infos):
        depth = np.array(Image.open(f'{os.path.dirname(cam_info.image_path)}/depth/{cam_info.image_name}_output.png')) / 256
        # depth = 20 * depth #room 5 # flower 20
        im_data = np.array(cam_info.image, dtype=np.float32) #/ 255.0

        disparity = 1/depth
        vis_photos, vis_depths = sparse_bilateral_filtering(disparity.copy(), im_data.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
        disparity = vis_depths[-1]
        depth_filtered = 1/disparity

        # depth = 20 * depth

        depth_diff = np.abs(depth_filtered - depth)
        trp_indices = np.argwhere(depth_diff.reshape(-1) > 0.1)[:, 0]

        depth = depth_filtered

        ### plane masks ###

        # depth_filtered = (depth_filtered - depth_filtered.min()) / (depth_filtered.max() - depth_filtered.min())
        # depth_filtered = np.clip(depth_filtered, 0.01, 0.99)

        # dep_masks = []
        # num_planes = 10
        # for i in range(num_planes):
        #     multp = 1.0 / num_planes
        #     dep_copy = depth_filtered.copy()
        #     dep_copy[depth_filtered <= (i * multp)] = 0
        #     dep_copy[depth_filtered > ((i + 1) * multp)] = 0
        #     dep_copy[dep_copy > 0] = 1

        #     dep_masks.append(dep_copy[None, None])

        # dep_masks = torch.tensor(np.concatenate(dep_masks, axis=1))
        dep_masks = torch.tensor(np.load(f'{os.path.dirname(cam_info.image_path)}/masks/{cam_info.image_name.split(".")[0]}.npy')).permute(2, 0, 1)[None]
        # dep_masks = torch.ones_like(dep_masks)

        # trp_indices = np.array([0])


        depth = torch.Tensor(depth)[None, None]

        focal = .5 * width / np.tan(.5 * float(cam_info.FovX))
        # print(focal)

        # Init radius equal to shorter length of the rectangle. Default: Height
        fovy = focal2fov(fov2focal(cam_info.FovX, width), height)
        # print(fovy)
        # Radii per frame
        radii = np.tan(0.5 * float(fovy))  * depth / height
        radii2 = radii**2

        K = torch.eye(3)[None]
        K[:, 0, 0] = focal
        K[:, 0, 2] = width / 2.0
        K[:, 1, 1] = focal#fov2focal(focal2fov(fov2focal(cam_info.FovX, width), height), height)
        K[:, 1, 2] = height / 2.0
        # print(depth.max(), depth.min(), K)
        camera3d = depth_to_3d(depth, K)

        # print(depth.shape, im_data.shape, radii2.shape, camera3d.shape)
        # exit()


        xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3).numpy()
        rgb = torch.Tensor(im_data).reshape(-1, 3).numpy()

        radii2 = radii2[0].permute(1, 2, 0).reshape(-1).numpy()

        indices = np.arange(width * height)
        xyz_cam_valid = xyz_cam[indices]
        rgb_valid = rgb[indices][..., :3]
        radii2_valid = radii2[indices][..., None]

        xyz_arr.append(xyz_cam_valid)
        rgb_arr.append(rgb_valid)
        radii2_arr.append(radii2_valid)
        cam_idx_arr.append(np.repeat(idx, xyz_cam_valid.shape[0]))
        indices_arr.append(torch.tensor(trp_indices + idx*width*height).cuda())
        depth_arr.append(depth)
        dep_mask_arr.append(dep_masks)

    dep_mask_arr = torch.cat(dep_mask_arr, dim=0).float().cuda()

    image_names = [cam_info.image_name.split('.')[0] for cam_info in train_cam_infos]
    print(image_names)
    flows, masks = {}, {}
    for idx, im1 in enumerate(image_names):
        for idx2, im2 in enumerate(image_names):
            if idx == idx2:
                continue

            flows[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}.npy")).cuda()
            masks[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}w.npy")).cuda()

    # inddd = 0
    # xyz = xyz_arr[inddd]
    # rgb = rgb_arr[inddd]
    # radii2 = radii2_arr[inddd]
    # cam_idx = cam_idx_arr[inddd]
    # indices_arr = indices_arr[inddd]
    # depth = depth_arr[inddd]

    xyz = np.concatenate(xyz_arr, axis=0)
    rgb = np.concatenate(rgb_arr, axis=0)
    radii2 = np.concatenate(radii2_arr, axis=0)
    cam_idx = np.concatenate(cam_idx_arr, axis=0)
    indices_arr = torch.cat(indices_arr, dim=0)
    depth = torch.cat(depth_arr, dim=1)


    ply_path = os.path.join(path, "points3d.ply")

    if os.path.exists(ply_path):
        os.remove(ply_path)


    storePly(ply_path, xyz, rgb, np.zeros_like(xyz), radii2, cam_idx[..., None])
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, indices_arr, depth, flows, masks, dep_mask_arr


sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    "Colmap": readLLFFInfo,
    "Blender" : readNerfSyntheticInfo,
    "DTU" : readDTUInfo,
    # "LLFF" : readLLFFInfo
}
