import sys
import os

sys.path.append(os.getcwd())
from tqdm import tqdm

import subprocess
import zipfile
import pandas as pd
import numpy as np
from utilities import find_files_with_string
from path_constants import ABLATION_PARAMETERS_CSV

#from sklearn.covariance import EllipticEnvelope
#from Evaluate.metrics import recall_ate

def evo_ape_zip(groundtruth_file, trajectory_file, evaluation_folder, max_time_difference=0.1):
    traj_file_name = os.path.basename(trajectory_file).replace(".txt", "")
    traj_zip = os.path.join(evaluation_folder, f"{traj_file_name}.zip")
    traj_tum = os.path.join(evaluation_folder, f"{traj_file_name}.tum")

    if os.path.exists(traj_tum):
        return

    command = (f"evo_ape tum {groundtruth_file} {trajectory_file} -va -as "
               f"--t_max_diff {max_time_difference} --save_results {traj_zip}")
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if not os.path.exists(traj_zip):
        return

    with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(trajectory_file + '.tum'):
                with zip_ref.open(file_name) as source_file:
                    destination_file_path = os.path.join(evaluation_folder,
                                                         os.path.basename(file_name).replace(".txt", ""))
                    with open(destination_file_path, 'wb') as target_file:
                        target_file.write(source_file.read())
                break

    df = pd.read_csv(destination_file_path, delimiter=' ', header=None)
    df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    df = df.sort_values(by='ts')
    df.to_csv(destination_file_path, index=False)

    gt_file = os.path.join(evaluation_folder, 'gt.tum')
    if os.path.exists(gt_file):
        return

    with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(groundtruth_file + '.tum'):
                with zip_ref.open(file_name) as source_file:
                    destination_file_path = gt_file
                    with open(destination_file_path, 'wb') as target_file:
                        target_file.write(source_file.read())
                break

    df = pd.read_csv(gt_file, delimiter=' ', header=None)
    df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    df = df.sort_values(by='ts')
    df.to_csv(gt_file, index=False)


def evo_get_accuracy(evaluation_folder):
    # Append new data to accuracy_raw
    accuracy_raw = os.path.join(evaluation_folder, 'accuracy_raw.csv')
    if os.path.exists(accuracy_raw):
        existing_data = pd.read_csv(accuracy_raw)
        os.remove(accuracy_raw)
    else:
        existing_data = None

    command = (f"pixi run -e evo evo_res {os.path.join(evaluation_folder, "*.zip")} --save_table {accuracy_raw}")
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if os.path.exists(accuracy_raw):
        new_data = pd.read_csv(accuracy_raw)
        if existing_data is not None:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.to_csv(accuracy_raw, index=False)
    else:
        if existing_data is not None:
            existing_data.to_csv(accuracy_raw, index=False)

    #
    df = pd.read_csv(accuracy_raw)
    df = df.rename(columns={df.columns[0]: 'traj_name'})
    data = df['rmse'].dropna()
    data_reshaped = data.values.reshape(-1, 1)

    # Number of Evaluation Points and number of Estimated Frames
    keyframe_traj_files = [f for f in os.listdir(evaluation_folder) if '_KeyFrameTrajectory.tum' in f]
    keyframe_traj_files.sort()
    for keyframe_traj_file in keyframe_traj_files:
        with open(os.path.join(evaluation_folder, keyframe_traj_file), 'r') as file:
            traj_name = keyframe_traj_file.replace('.tum', '.txt')
            num_evaluation_points = sum(1 for line in file) - 1
            df.loc[df['traj_name'] == traj_name, 'Number of Evaluation Points'] = num_evaluation_points
            keyframe_traj_file_not_aligned = os.path.join(evaluation_folder, '..', traj_name)
        with open(keyframe_traj_file_not_aligned, 'r') as file:
            num_estimated_frames = sum(1 for line in file)
            df.loc[df['traj_name'] == traj_name, 'Number of Estimated Frames'] = num_estimated_frames

        #traj = pd.read_csv(os.path.join(evaluation_folder, keyframe_traj_file))
        #gt = pd.read_csv(os.path.join(evaluation_folder, 'gt.tum'))

        #traj_xyz = traj[['tx', 'ty', 'tz']]
        #gt_xyz = gt[['tx', 'ty', 'tz']]

    # Use EllipticEnvelope to fit the data
    num_traj_files = len(keyframe_traj_files)
    # if num_traj_files > 5000000:
    #     outlier_detector = EllipticEnvelope(contamination=0.10)  # 5% contamination is typical
    #     outliers = outlier_detector.fit_predict(data_reshaped)
    #     outlier_indices = np.where(outliers == -1)[0]
    #     cleaned_df = df.drop(index=outlier_indices)
    # else:
    #     outlier_indices = []
    #     cleaned_df = df
    outlier_indices = []
    cleaned_df = df

    accuracy = os.path.join(evaluation_folder, 'accuracy.csv')
    if os.path.exists(accuracy):
        os.remove(accuracy)
    cleaned_df.to_csv(accuracy, index=False)

    outlier_file_names = df.iloc[outlier_indices].iloc[:, 0]
    for outlier_file_name in outlier_file_names:
        traj_file = os.path.join(evaluation_folder, f"{outlier_file_name.replace(".txt", "")}.tum")
        if (os.path.exists(traj_file)):
            os.remove(traj_file)

def find_groundtruth_txt(trajectories_path, trajectory_file, parameter):
    ablation_parameters_csv = os.path.join(trajectories_path, ABLATION_PARAMETERS_CSV)
    traj_name = os.path.basename(trajectory_file)
    df = pd.read_csv(ablation_parameters_csv)
    index_str = traj_name.split('_')[0]
    expId = int(index_str)
    exp_row = df[df['expId'] == expId]
    ablation_values = exp_row[parameter].values[0]

    min_noise = df['std_noise'].min()
    df_noise_filter = df[df['std_noise'] == min_noise]
    gt_ids = df_noise_filter[(df_noise_filter[parameter].sub(ablation_values).abs() == df_noise_filter[parameter].sub(
        ablation_values).abs().min())]

    gt_id = expId
    while gt_id == expId:
        gt_id = np.random.choice(gt_ids['expId'].values)

    groundtruth_txt = os.path.join(trajectories_path, f"{str(gt_id).zfill(5)}_KeyFrameTrajectory.txt")
    return groundtruth_txt

if __name__ == "__main__":
    if len(sys.argv) > 2:
        function_name = sys.argv[1]
        max_time_difference = sys.argv[2]
        trajectories_path = sys.argv[3]
        evaluation_folder = sys.argv[4]
        groundtruth_file = sys.argv[5]
        psudo_groundtruth = bool(int(sys.argv[6]))

        trajectory_files = find_files_with_string(trajectories_path, "_KeyFrameTrajectory.txt")
        if function_name == "evo_ape_zip":
            for trajectory_file in tqdm(trajectory_files):
                if psudo_groundtruth:
                    parameter = sys.argv[7]
                    groundtruth_file = find_groundtruth_txt(trajectories_path, trajectory_file, parameter)
                evo_ape_zip(groundtruth_file, trajectory_file, evaluation_folder, float(max_time_difference))
