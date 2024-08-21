#!/bin/bash

DIVISOR="10"
MAX_RGB="50"

# Function to split key-value pairs and assign them to variables
split_and_assign() {
  local input=$1
  local key=$(echo $input | cut -d':' -f1)
  local value=$(echo $input | cut -d':' -f2-)
  eval $key=$value
}


# Split the input string into individual components
split_and_assign "$1"
split_and_assign "$2"
split_and_assign "$3"
split_and_assign "$4"

echo "Sequence Path: $sequence_path"
echo "Experiment Folder: $exp_folder"
echo "Experiment ID: $exp_id"
echo "Verbose: $verbose"

exp_id=$(printf "%05d" ${exp_id})

# Calculate the minimum frames per second (fps) for downsampling
calibration_file="${sequence_path}/calibration.yaml"
fps=$(grep -oP '(?<=Camera\.fps:\s)-?\d+\.\d+' "$calibration_file")
min_fps=$(echo "scale=2; $fps / ${DIVISOR}" | bc)

exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rm -rf "$exp_folder_colmap"
mkdir "$exp_folder_colmap"

# Downsample RGB frames
rgb_ds_txt="${exp_folder_colmap}/rgb_ds.txt"
python snippets/downsample_rgb_frames.py $sequence_path --rgb_ds_txt "${rgb_ds_txt}" --min_fps ${min_fps} -v --max_rgb ${MAX_RGB}

# Run COLMAP scripts for matching and mapping
./VSLAM-Baselines/COLMAP/colmap_matcher.sh $sequence_path $exp_folder $exp_id 
./VSLAM-Baselines/COLMAP/colmap_mapper.sh $sequence_path $exp_folder $exp_id 

# Convert COLMAP outputs to a format suitable for VSLAM-Lab
python VSLAM-Baselines/COLMAP/colmap_to_vslamlab.py $sequence_path $exp_folder $exp_id $verbose


