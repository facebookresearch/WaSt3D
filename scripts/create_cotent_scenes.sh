#!/bin/bash

# List of strings to iterate over
strings=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship" )
# strings=("chair" "drums" "ficus" "hotdog"  "materials" "mic" "ship" )

strings=("chair" "drums" "ficus" "materials" "mic" "ship" )

# # Loop through the strings
# for string in "${strings[@]}"; do
#     CUDA_VISIBLE_DEVICES=1
#      python train_spheres.py \
#         -s ~/data/nerf_synthetic/${string}/ \
#         --model_path ~/output_data/gs_my/gs/new_scenes/GT_${string}_spheres \
#         --port 6001
# done





# For NERF synthetic dataset
# strings=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship" )

# # Loop through the strings
# for string in "${strings[@]}"; do
#     CUDA_VISIBLE_DEVICES=2 python train.py \
#         -s ~/data/nerf_synthetic/${string}/ \
#         --model_path ~/output_data/gs/GT_${string} \
#         --port 6001
# done


# For NERF_LLFF dataset
strings=("fern"  "flower"  "fortress"  "horns"  "house"  "leaves"  "livingroom"
        "orchids"  "room"  "trex"  "xmaschair")
DEV=3

# Loop through the strings
mkdir ~/output_data/gs/content/llff/
for string in "${strings[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV python train.py \
        -s ~/data/nerf_llff/${string}/ \
        --model_path ~/output_data/gs/content/llff/GT_${string} \
        --port 600$DEV
done
