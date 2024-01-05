# Launch training with fitting spheres with very spherical sizes
CUDA_VISIBLE_DEVICES=1 with-proxy  python train_spheres.py \
    -s ~/data/nerf_synthetic/hotdog/ \
    --model_path ~/output_data/gs_my/gs/GT_hotdog_spheres_v1 \
    --port 6001

CUDA_VISIBLE_DEVICES=1 with-proxy  python train_spheres.py \
    -s ~/data/nerf_synthetic/lego/ \
    --model_path ~/output_data/gs_my/gs/GT_lego_spheres \
    --port 6001

CUDA_VISIBLE_DEVICES=1 with-proxy  python train.py \
    -s ~/data/nerf_synthetic/lego/ \
    --model_path ~/output_data/gs_my/gs/GT_lego \
    --port 6001

# Start nerfstudio viewer
python nerfstudio/scripts/gaussian_splatting/run_viewer.py \
    --model-path ~/workspace/gaussian-splatting/notebooks/GT_lego_cluster2/ \
    --config.viewer.websocket-port 42915

python nerfstudio/scripts/gaussian_splatting/run_viewer.py \
    --model-path ~/output_data/gs_my/gs/GT_lego \
    --config.viewer.websocket-port 42915

# Generate nerfstudion video
python nerfstudio/scripts/gaussian_splatting/render.py camera-path \
    --model-path ~/workspace/gaussian-splatting/notebooks/GT_lego_cluster2/ \
    --camera-path-filename ../gaussian-splatting/scripts/camera_path_hotdog.json \
    --output-path ~/output_data/gs_videos/GT_lego_nonoverlapping_patches.mp4

# Generate video in high resolution
CUDA_VISIBLE_DEVICES=2 python nerfstudio/scripts/gaussian_splatting/render.py camera-path \
    --model-path ~/workspace/gaussian-splatting/notebooks/fit_hotdog_with_lego_shaded/ \
    --camera-path-filename ../gaussian-splatting/scripts/camera_path_hotdog.json \
    --output-path ~/output_data/gs_videos/fit_hotdog_with_lego_shaded_hires_2K.mp4 \
    --downscale-factor 0.4

# Partition into clusters

CUDA_VISIBLE_DEVICES=1 with-proxy  python train.py \
    -s ~/data/nerf_synthetic/lego/ \
    --model_path ~/output_data/gs_my/gs/GT_lego \
    --port 6001
