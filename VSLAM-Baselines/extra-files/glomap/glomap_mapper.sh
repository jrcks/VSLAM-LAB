#!/bin/bash
echo "Executing glomapMapper.sh ..."

sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 
verbose="$4"
settings_yaml="$5"

rgb_path="${sequence_path}/rgb"
exp_folder_colmap="${exp_folder}/colmap_${exp_id}"


# Reading settings from yaml file
BundleAdjustment_thres_loss_function=$(yq '.BundleAdjustment.thres_loss_function // 1.0' $settings_yaml)
RelPoseEstimation_max_epipolar_error=$(yq '.RelPoseEstimation.max_epipolar_error // 1.0' $settings_yaml)
GlobalPositioning_thres_loss_function=$(yq '.GlobalPositioning.thres_loss_function // 0.10000000000000001' $settings_yaml)

echo "    glomap mapper ..."
database="${exp_folder_colmap}/colmap_database.db"

pixi run -e colmap glomap mapper \
    --database_path ${database} \
    --image_path ${rgb_path} \
    --output_path ${exp_folder_colmap} \
    --skip_view_graph_calibration 1 \
    --RelPoseEstimation.max_epipolar_error "${RelPoseEstimation_max_epipolar_error}" \
    --GlobalPositioning.thres_loss_function "${GlobalPositioning_thres_loss_function}" \
    --BundleAdjustment.optimize_intrinsics 0 \
    --BundleAdjustment.thres_loss_function "${BundleAdjustment_thres_loss_function}"

if [ "$verbose" -eq 1 ]; then
  pixi run -e colmap colmap gui --import_path "${exp_folder_colmap}/0" --database_path ${database} --image_path ${rgb_path}
fi

echo "    colmap model_converter ..."
pixi run -e colmap colmap model_converter \
	--input_path ${exp_folder_colmap}/0 --output_path ${exp_folder_colmap} --output_type TXT


