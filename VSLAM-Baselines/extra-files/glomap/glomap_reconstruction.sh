#!/bin/bash

DIVISOR="10"

# Function to split key-value pairs and assign them to variables
max_rgb="50" # Default value. Can be overwritten "max_rgb:500"
matcher_type="exhaustive" # Default value. Options: exhaustive, sequential
use_gpu="1" # Default value.
verbose="0"
settings_yaml=" VSLAM-Baselines/glomap/glomap_settings.yaml"

split_and_assign() {
  local input=$1
  local key=$(echo $input | cut -d':' -f1)
  local value=$(echo $input | cut -d':' -f2-)
  eval $key=$value
}

# Split the input string into individual components
for ((i=1; i<=$#; i++)); do
    split_and_assign "${!i}"
done

exp_id=$(printf "%05d" ${exp_id})

echo "Sequence Path: $sequence_path"
echo "Experiment Folder: $exp_folder"
echo "Experiment ID: $exp_id"
echo "Verbose: $verbose"
echo "max_rgb: $max_rgb"
echo "matcher_type: $matcher_type"
echo "use_gpu: $use_gpu"
echo "settings_yaml: $settings_yaml"

# Calculate the minimum frames per second (fps) for downsampling
calibration_file="${sequence_path}/calibration.yaml"
fps=$(grep -oP '(?<=Camera\.fps:\s)-?\d+\.\d+' "$calibration_file")
min_fps=$(echo "scale=2; $fps / ${DIVISOR}" | bc)

exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rm -rf "$exp_folder_colmap"
mkdir "$exp_folder_colmap"

# Downsample RGB frames
rgb_ds_txt_0="${sequence_path}/rgb_ds.txt"
rgb_ds_txt="${exp_folder_colmap}/rgb_ds.txt"

if [ ! -f "${rgb_ds_txt_0}" ]; then
  python snippets/downsample_rgb_frames.py "${sequence_path}" --rgb_ds_txt "${rgb_ds_txt}" --min_fps "${min_fps}" -v --max_rgb "${max_rgb}"
else
  cp "${rgb_ds_txt_0}" "${rgb_ds_txt}"
fi

# Run COLMAP scripts for matching and mapping
pixi run -e colmap ./VSLAM-Baselines/glomap/glomap_matcher.sh $sequence_path $exp_folder $exp_id $matcher_type $use_gpu ${settings_yaml}
pixi run -e colmap ./VSLAM-Baselines/glomap/glomap_mapper.sh $sequence_path $exp_folder $exp_id ${verbose} ${settings_yaml}

# Convert COLMAP outputs to a format suitable for VSLAM-Lab
python VSLAM-Baselines/colmap/colmap_to_vslamlab.py $sequence_path $exp_folder $exp_id $verbose


