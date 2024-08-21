#!/bin/bash
echo "Executing colmapMatcher.sh ..."

calibration_model="FULL_OPENCV" # PINHOLE, FULL_OPENCV, OPENCV_FISHEYE
sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 

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
colmap database_creator --database_path ${database} 

################################################################################	
echo "    colmap feature_extractor ..."
if [ "${calibration_model}" == "FULL_OPENCV" ] 
then
	colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}, ${k1}, ${k2}, ${p1}, ${p2}, ${k3}, ${k4}, ${k5}, ${k6}"
fi

if [ "${calibration_model}" == "FULL_OPENCV" ] 
then
	colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \	
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}, ${k1}, ${k2}, ${p1}, ${p2}, ${k3}, ${k4}, ${k5}, ${k6}"
fi

if [ "${calibration_model}" == "OPENCV_FISHEYE" ] 
then
	colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \	
	--ImageReader.camera_model ${calibration_model} \
	--ImageReader.single_camera 1 \
	--ImageReader.camera_params "${fx}, ${fy}, ${cx}, ${cy}, ${k1}, ${k2}, ${k3}, ${k4}" 
fi

################################################################################	
echo "    colmap exhaustive_matcher ..."	
colmap exhaustive_matcher \
   --database_path ${database} 
