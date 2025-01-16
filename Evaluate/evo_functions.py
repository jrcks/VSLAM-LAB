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
import re


#from sklearn.covariance import EllipticEnvelope
#from Evaluate.metrics import recall_ate

def evo_metric(metric, groundtruth_file, trajectory_file, evaluation_folder, max_time_difference=0.1, index="none"):
    accuracy_csv = os.path.join(evaluation_folder, f"{metric}.csv")
    traj_file_name = os.path.basename(trajectory_file).replace(".txt", "")

    trajectory = pd.read_csv(trajectory_file, delimiter=' ', header=None)
    trajectory_sorted = trajectory.sort_values(by=trajectory.columns[0])

    #last_column = trajectory_sorted.iloc[:, -1]
    #trajectory_sorted = trajectory_sorted.iloc[:, :-1]
    #trajectory_sorted.insert(4, 'last_column', last_column)
    if index != "none":
        traj_file_name = traj_file_name.replace("KeyFrameTrajectory", f"{index:02d}_KeyFrameTrajectory")
        trajectory_file = os.path.join(evaluation_folder, f"{traj_file_name}.txt")

    trajectory_sorted.to_csv(trajectory_file, header=None, index=False, sep=' ', lineterminator='\n')

    # Check if evaluation already exists
    if os.path.exists(accuracy_csv):
        accuracy_file = pd.read_csv(accuracy_csv)
        if os.path.basename(trajectory_file) in accuracy_file['traj_name'].values:
            return

    traj_zip = os.path.join(evaluation_folder, f"{traj_file_name}.zip")
    traj_tum = os.path.join(evaluation_folder, f"{traj_file_name}.tum")
    gt_tum = traj_tum.replace("KeyFrameTrajectory", "gt")

    if os.path.exists(traj_zip):
        return

    # Evaluate
    if metric == 'ate':
        command = (f"evo_ape tum {groundtruth_file} {trajectory_file} -va -as "
                   f"--t_max_diff {max_time_difference} --save_results {traj_zip}")
    if metric == 'rpe':
        command = f"evo_rpe tum {groundtruth_file} {trajectory_file} --all_pairs --delta 5 -va -as --save_results {traj_zip}"

    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if not os.path.exists(traj_zip):
        return

    # Write aligned trajectory
    if os.path.exists(traj_tum):
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
    #df.to_csv(destination_file_path, header=None, index=False, sep=' ', lineterminator='\n')
    df.to_csv(destination_file_path, index=False, sep=' ', lineterminator='\n')

    # Write aligned gt
    if os.path.exists(gt_tum):
        return

    with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(groundtruth_file + '.tum'):
                with zip_ref.open(file_name) as source_file:
                    with open(gt_tum, 'wb') as target_file:
                        target_file.write(source_file.read())
                break

    df = pd.read_csv(gt_tum, delimiter=' ', header=None)
    df.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    df = df.sort_values(by='ts')
    #df.to_csv(gt_tum, header=None, index=False, sep=' ', lineterminator='\n')
    df.to_csv(gt_tum, index=False, sep=' ', lineterminator='\n')


def evo_get_accuracy(metric, evaluation_folder):

    files_in_folder = os.listdir(evaluation_folder)
    zip_files = [file for file in files_in_folder if file.endswith('.zip')]
    zip_files.sort()
    num_zip_files = len(zip_files)
    if num_zip_files == 0:
        return
    chunk_size = 500
    zip_files_chunks = [zip_files[i:i + chunk_size] for i in range(0, len(zip_files), chunk_size)]
    zip_files_chunks = [' '.join(os.path.join(evaluation_folder, file) for file in chunk) for chunk in zip_files_chunks]

    accuracy_raw = os.path.join(evaluation_folder, f'{metric}_raw.csv')
    for zip_file_chunk in zip_files_chunks:
        if os.path.exists(accuracy_raw):
            existing_data = pd.read_csv(accuracy_raw)
            os.remove(accuracy_raw)
        else:
            existing_data = None

        command = (f"pixi run -e evo evo_res {zip_file_chunk} --save_table {accuracy_raw}")
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

    # Number of Evaluation Points and number of Estimated Frames
    keyframe_traj_files = [f for f in os.listdir(evaluation_folder) if '_KeyFrameTrajectory.tum' in f]
    keyframe_traj_files.sort()
    for keyframe_traj_file in keyframe_traj_files:
        with open(os.path.join(evaluation_folder, keyframe_traj_file), 'r') as file:
            traj_name = keyframe_traj_file.replace('.tum', '.txt')
            num_evaluation_points = sum(1 for line in file) - 1
            df.loc[df['traj_name'] == traj_name, 'Number of Evaluation Points'] = num_evaluation_points

            traj_name_0 = re.sub(r"_(\d{2})_", "_", traj_name)
            keyframe_traj_file_not_aligned = os.path.join(evaluation_folder, '..', traj_name_0)
        with open(keyframe_traj_file_not_aligned, 'r') as file:
            num_estimated_frames = sum(1 for line in file)
            df.loc[df['traj_name'] == traj_name, 'Number of Estimated Frames'] = num_estimated_frames

    accuracy = os.path.join(evaluation_folder, f'{metric}.csv')
    if os.path.exists(accuracy):
        os.remove(accuracy)
    df.to_csv(accuracy, index=False)

    # Remove zip files
    for filename in os.listdir(evaluation_folder):
        if filename.endswith('.zip'):
            file_path = os.path.join(evaluation_folder, filename)
            os.remove(file_path)


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

    threshold_percent = 0.1
    lower_bound = ablation_values * (1 - threshold_percent / 100)
    upper_bound = ablation_values * (1 + threshold_percent / 100)

    gt_ids = df_noise_filter[
        (df_noise_filter[parameter] >= lower_bound) & (df_noise_filter[parameter] <= upper_bound)
        ]
    groundtruths_txt = []
    for gt_id in gt_ids['expId'].values:
        groundtruth_txt = os.path.join(trajectories_path, f"{str(gt_id).zfill(5)}_KeyFrameTrajectory.txt")
        if gt_id != expId:
            if os.path.exists(groundtruth_txt):
                groundtruths_txt.append(groundtruth_txt)

    return groundtruths_txt


def compute_trajectory_length(trajectory_file):
    df = pd.read_csv(trajectory_file, usecols=['tx', 'ty', 'tz'], delimiter=' ')
    data = df.to_numpy()
    distances = np.linalg.norm(np.diff(data, axis=0), axis=1)
    trajectory_length = np.sum(distances)
    return trajectory_length


def compute_trajectory_lengths(evaluation_folder, metric):
    csv_file = os.path.join(evaluation_folder, f'{metric}.csv')
    df = pd.read_csv(csv_file)
    trajectory_lengths = []
    for traj_name in df['traj_name']:
        traj_txt = os.path.join(evaluation_folder, traj_name)
        traj_tum = traj_txt.replace('.txt', '.tum')
        if os.path.exists(traj_tum):
            length = compute_trajectory_length(traj_tum)
            trajectory_lengths.append(length)
        else:
            trajectory_lengths.append(None)
    df['trajectory_length'] = trajectory_lengths
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        function_name = sys.argv[1]
        max_time_difference = sys.argv[2]
        trajectories_path = sys.argv[3]
        evaluation_folder = sys.argv[4]
        groundtruth_file = sys.argv[5]
        pseudo_groundtruth = bool(int(sys.argv[6]))

        trajectory_files = find_files_with_string(trajectories_path, "_KeyFrameTrajectory.txt")
        if function_name == "ate" or function_name == "rpe":
            for trajectory_file in tqdm(trajectory_files):
                if pseudo_groundtruth:
                    parameter = sys.argv[7]
                    groundtruth_files = find_groundtruth_txt(trajectories_path, trajectory_file, parameter)
                    for idx, groundtruth_file in enumerate(groundtruth_files):
                        evo_metric(function_name, groundtruth_file, trajectory_file, evaluation_folder,
                                   float(max_time_difference), idx)
                else:
                    evo_metric(function_name, groundtruth_file, trajectory_file, evaluation_folder,
                               float(max_time_difference))
            evo_get_accuracy(function_name, evaluation_folder)
            compute_trajectory_lengths(evaluation_folder, function_name)
