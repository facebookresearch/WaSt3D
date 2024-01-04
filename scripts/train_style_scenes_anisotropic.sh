#!/bin/bash

# List of strings to iterate over
strings=("bark" "brickwall"  "bunny_plush"  "carpet_camouflage"  "carpet_long"  "coast_land_rocks"  "gazon"  "rocks"  "sticks_and_logs"  "tree"  "wicker_basket" "skull" "rose_bush")
strings=("skull" "rose_bush")
strings=("rose_bush_hires")
strings=("pine_cone")
strings=("cotoneaster2")

# strings=("anthurium pine_cone rose_bush_hires skull pebbles")

strings=("anthurium" "skull" "pebbles" "sticks_and_logs"  "rose_bush_hires" "rose_bush")
# strings=("rose_bush")

# strings=("chair" "drums" "ficus" "hotdog"  "materials" "mic" "ship" )
DEV=1
INPUT_DIR=~/data/style_scenes/
OUTPUT_DIR=~/output_data/gs_my/style_scenes_spheres_anisotropic/
mkdir $OUTPUT_DIR
# Loop through the strings
for string in "${strings[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV python train_spheres_anisotropic.py \
        -s ${INPUT_DIR}/${string}/ \
        --model_path ${OUTPUT_DIR}/GT_${string}_spheres \
        --port 6001 #\
        #--resolution 2048
done


# CUDA_VISIBLE_DEVICES=3 python train_spheres_anisotropic.py \
#         -s ~/data/style_scenes/anthurium \
#         --model_path ~/output_data/gs_my/style_scenes_spheres_anisotropic/GT_anthurium_spheres \
#         --port 6003



# CUDA_VISIBLE_DEVICES=3 python train_spheres_anisotropic_simple.py \
#         -s ~/data/style_scenes/anthurium \
#         --model_path ~/output_data/gs_my/style_scenes_spheres_anisotropic/GT_anthurium_spheres_simple \
#         --port 6003
