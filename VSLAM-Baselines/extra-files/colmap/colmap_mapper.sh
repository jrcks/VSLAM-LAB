#!/bin/bash
echo "Executing colmapMapper.sh ..."

sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 
verbose="$4"

rgb_path="${sequence_path}/rgb"
exp_folder_colmap="${exp_folder}/colmap_${exp_id}"

echo "    colmap mapper ..."
database="${exp_folder_colmap}/colmap_database.db"
	
pixi run -e colmap colmap mapper \
    --database_path ${database} \
    --image_path ${rgb_path} \
    --output_path ${exp_folder_colmap} \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 
    
    #--Mapper.init_max_error ${reprojError} \
    #--Mapper.filter_max_reproj_error ${reprojError} \
    #--Mapper.tri_merge_max_reproj_error ${reprojError} \
    #--Mapper.tri_complete_max_reproj_error ${reprojError} \

n=5 
for ((i=1; i<=n; i++)); do
    echo "    colmap bundle_adjuster ${i} ..."	
    pixi run -e colmap colmap bundle_adjuster \
	--input_path ${exp_folder_colmap}/0 \
	--output_path ${exp_folder_colmap}/0 \
	--BundleAdjustment.refine_focal_length 0 \
	--BundleAdjustment.refine_principal_point 0 
done    

echo "    colmap model_converter ..."	
pixi run -e colmap colmap model_converter \
	--input_path ${exp_folder_colmap}/0 --output_path ${exp_folder_colmap} --output_type TXT	        

if [ "$verbose" -eq 1 ]; then
  pixi run -e colmap colmap gui --import_path "${exp_folder_colmap}/0" --database_path ${database} --image_path ${rgb_path}
fi

