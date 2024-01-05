#!/bin/bash
# Used this tutorial to process tuples of stings into individual arguments


DEV=1
INPUT_DIR=/home/dimakot55/output_data/gs_my/style_scenes_spheres_new/
OUTPUT_DIR=/home/dimakot55/output_data/style_clusters/
# Scenes names and number of clusters we need for scene
names_clusters=(
    "GT_rose_bush_spheres 10"
    "GT_rose_bush_spheres 5"
    "GT_skull_spheres 1"
    "GT_grass_spheres 3"
    "GT_bark_fragment_spheres 5"
    "GT_rocks4_spheres 6"
    "GT_anthurium_spheres 1"
    "GT_pebbles_spheres 5")

names_clusters=(
    "GT_grass_spheres 3"
    "GT_rocks4_spheres 6"
    "GT_pine_cone_spheres 1"
    "GT_wicker_basket_spheres 10"
    "GT_cotoneaster2_spheres 6"
    "GT_bark_spheres 6"
    "GT_brickwall_spheres 9"
    )


names_clusters=(
    #"GT_rocks4_spheres 10"
    "GT_pebbles_spheres 10"
    )

names_clusters=(
    "GT_grass_spheres 10"
    "GT_anthurium_spheres 10"
)

mkdir $OUTPUT_DIR
# Loop through the strings
for name_clusters in "${names_clusters[@]}"; do
    read -a vals <<< "$name_clusters"
    ckpt_path=${INPUT_DIR}\/${vals[0]}\/chkpnt30000.pth
    output_dir=${OUTPUT_DIR}\/${vals[0]}_${vals[1]}_clusters/
    num_clusters=${vals[1]}
    echo Start clustering ${ckpt_path} into  ${num_clusters} clusters

    CUDA_VISIBLE_DEVICES=$DEV python aux_save_clusters_clean.py \
        --ckpt_path ${ckpt_path} \
        --output_dir ${output_dir} \
        --num_clusters ${num_clusters}
    echo Done. Results are in ${output_dir}
done
