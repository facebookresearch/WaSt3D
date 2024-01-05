#!/bin/bash

# List of strings to iterate over
strings=("bark" "brickwall"  "bunny_plush"  "carpet_camouflage"  "carpet_long"  "coast_land_rocks"  "gazon"  "rocks"  "sticks_and_logs"  "tree"  "wicker_basket" "skull" "rose_bush")
strings=("skull" "rose_bush")
strings=("rose_bush_hires")
strings=("pine_cone")
strings=("cotoneaster2")

# strings=("anthurium pine_cone rose_bush_hires skull pebbles")

strings=("skull" "pebbles" "rose_bush" "bark_fragment" "sticks_and_logs" "anthurium" "rose_bush_hires")

strings=("rocks4" "pine_cone" "wicker_basket" "cotoneaster2" "bark" "brickwall")
strings=("grass")

# strings=("rose_bush")

# strings=("chair" "drums" "ficus" "hotdog"  "materials" "mic" "ship" )
DEV=3
INPUT_DIR=~/data/style_scenes/
OUTPUT_DIR=~/output_data/gs_my/style_scenes_spheres_new/
mkdir $OUTPUT_DIR
# Loop through the strings
for string in "${strings[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV python train_spheres.py \
        -s ${INPUT_DIR}/${string}/ \
        --model_path ${OUTPUT_DIR}/GT_${string}_spheres \
        --port 600${DEV} #\
        #--resolution 2048
done
