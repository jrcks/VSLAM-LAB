#!/bin/bash
echo ""
echo "Executing colmapMatcher.sh ..."

calibration_model="OPENCV" # PINHOLE, OPENCV, OPENCV_FISHEYE
sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 
matcher_type="$4" # Options: exhaustive, sequential
use_gpu="$5"
settings_yaml="$6"

calibration_file="${sequence_path}/calibration.yaml"
rgb_path="${sequence_path}/rgb"
exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rgb_ds_txt="${exp_folder_colmap}/rgb_ds.txt"


fx=$(grep -oP '(?<=Camera\.fx:\s)-?\d+\.\d+' "$calibration_file")
fy=$(grep -oP '(?<=Camera\.fy:\s)-?\d+\.\d+' "$calibration_file")
cx=$(grep -oP '(?<=Camera\.cx:\s)-?\d+\.\d+' "$calibration_file")
cy=$(grep -oP '(?<=Camera\.cy:\s)-?\d+\.\d+' "$calibration_file")

k1=$(grep -oP '(?<=Camera\.k1:\s)-?\d+\.\d+' "$calibration_file")
k2=$(grep -oP '(?<=Camera\.k2:\s)-?\d+\.\d+' "$calibration_file")
p1=$(grep -oP '(?<=Camera\.p1:\s)-?\d+\.\d+' "$calibration_file")
p2=$(grep -oP '(?<=Camera\.p2:\s)-?\d+\.\d+' "$calibration_file")
k3=$(grep -oP '(?<=Camera\.k3:\s)-?\d+\.\d+' "$calibration_file")
k4=0.0
k5=0.0
k6=0.0

# Create colmap image list
colmap_image_list="${exp_folder_colmap}/colmap_image_list.txt"
awk '{print substr($2, 5)}' "$rgb_ds_txt" > "$colmap_image_list"

# Create Colmap Database
database="${exp_folder_colmap}/colmap_database.db"
rm -rf ${database}
pixi run -e colmap colmap database_creator --database_path ${database}

################################################################################
echo "    colmap feature_extractor ..."
if [ "${calibration_model}" == "PINHOLE" ]
then
  echo "        camera model : $calibration_model"
	pixi run -e colmap colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--SiftExtraction.use_gpu ${use_gpu} \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}"
fi

if [ "${calibration_model}" == "OPENCV" ]
then
  echo "        camera model : $calibration_model"
	pixi run -e colmap colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--SiftExtraction.use_gpu ${use_gpu} \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}, ${k1}, ${k2}, ${p1}, ${p2}"
fi

if [ "${calibration_model}" == "OPENCV_FISHEYE" ] 
then
  echo "        camera model : $calibration_model"
	pixi run -e colmap colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--SiftExtraction.use_gpu ${use_gpu} \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}, ${k1}, ${k2}, ${k3}, ${k4}" 
fi

################################################################################
if [ "${matcher_type}" == "exhaustive" ]
then
	echo "    colmap exhaustive_matcher ..."
  pixi run -e colmap colmap exhaustive_matcher \
     --database_path ${database} \
     --SiftMatching.use_gpu ${use_gpu}
fi

if [ "${matcher_type}" == "sequential" ]
then
  num_rgb=$(wc -l < ${rgb_ds_txt})

  # Pick vocabulary tree based on the number of images
  vocabulary_tree="VSLAM-Baselines/glomap/vocab_tree_flickr100K_words32K.bin"
  if [ "$num_rgb" -gt 1000 ]; then
    vocabulary_tree="VSLAM-Baselines/glomap/vocab_tree_flickr100K_words256K.bin"
  fi
  if [ "$num_rgb" -gt 10000 ]; then
    vocabulary_tree="VSLAM-Baselines/glomap/vocab_tree_flickr100K_words1M.bin"
  fi

  echo "    colmap sequential_matcher ..."
  echo "        Vocabulary Tree: $vocabulary_tree"
      pixi run -e colmap colmap sequential_matcher \
         --database_path ${database} \
         --SequentialMatching.loop_detection 1 \
         --SequentialMatching.vocab_tree_path ${vocabulary_tree} \
         --SiftMatching.use_gpu ${use_gpu}
fi
