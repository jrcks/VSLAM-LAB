#!/bin/bash
echo "Executing glomapMapper.sh ..."

sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 
verbose="$4"

rgb_path="${exp_folder}/rgb"
exp_folder_colmap="${exp_folder}/colmap_${exp_id}"

echo "    glomap mapper ..."
database="${exp_folder_colmap}/colmap_database.db"

pixi run -e colmap glomap mapper \
    --database_path ${database} \
    --image_path ${rgb_path} \
    --output_path ${exp_folder_colmap} \
    --skip_view_graph_calibration 1 \
    --BundleAdjustment.optimize_intrinsics 0

#n=5
#for ((i=1; i<=n; i++)); do
#    echo "    colmap bundle_adjuster ${i} ..."
#    pixi run -e colmap colmap bundle_adjuster \
#	--input_path ${exp_folder_colmap}/0 \
#	--output_path ${exp_folder_colmap}/0 \
#	--BundleAdjustment.refine_focal_length 0 \
#	--BundleAdjustment.refine_principal_point 0
#done

if [ "$verbose" -eq 1 ]; then
  pixi run -e colmap colmap gui --import_path "${exp_folder_colmap}/0" --database_path ${database} --image_path ${rgb_path}
fi

echo "    colmap model_converter ..."
pixi run -e colmap colmap model_converter \
	--input_path ${exp_folder_colmap}/0 --output_path ${exp_folder_colmap} --output_type TXT


