import subprocess
import os, shutil
import pandas as pd
from tqdm import tqdm

from Evaluate.evo_functions import evo_metric, evo_get_accuracy
from path_constants import VSLAM_LAB_EVALUATION_FOLDER, TRAJECTORY_FILE_NAME
from utilities import print_msg

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def evaluate_sequence(exp, dataset, sequence_name, overwrite=False):
    command =  "pixi run -e evo evo_config set save_traj_in_zip true"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    METRIC = 'ate'
    
    trajectories_path = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    groundtruth_file = os.path.join(sequence_path, 'groundtruth.txt')
    evaluation_folder = os.path.join(trajectories_path, VSLAM_LAB_EVALUATION_FOLDER)
    accuracy_csv = os.path.join(evaluation_folder, f'{METRIC}.csv')

    exp_log = pd.read_csv(exp.log_csv)
    if overwrite:
        if os.path.exists(evaluation_folder):
            shutil.rmtree(evaluation_folder)        
        exp_log.loc[exp_log["sequence_name"] == sequence_name, "EVALUATION"] = "none"

    os.makedirs(evaluation_folder, exist_ok=True)

    evaluated_runs = []
    for _, row in exp_log.iterrows():
        if row["SUCCESS"] and (row["EVALUATION"] == "none") and (row["sequence_name"] == sequence_name):
            exp_it = str(row["exp_it"]).zfill(5) 
            evaluated_runs.append(exp_it)

    print_msg(SCRIPT_LABEL, f"Evaluating '{evaluation_folder.replace(sequence_name, f"{dataset.dataset_color}{sequence_name}\033[0m")}'")
    if len(evaluated_runs) == 0:
        return
    
    zip_files = []
    for exp_it in tqdm(evaluated_runs):
        trajectory_file = os.path.join(trajectories_path, f"{exp_it}{TRAJECTORY_FILE_NAME}.txt")
        evo_metric('ate', groundtruth_file, trajectory_file, evaluation_folder, 1.0 / dataset.rgb_hz)
        zip_files.append(os.path.join(evaluation_folder, f"{exp_it}{TRAJECTORY_FILE_NAME}.zip"))

    evo_get_accuracy(zip_files, accuracy_csv)

    accuracy = pd.read_csv(accuracy_csv)
    for evaluated_run in evaluated_runs:
        exists = (accuracy["traj_name"] == f"{exp_it}{TRAJECTORY_FILE_NAME}.txt").any()
        if exists:
            exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = METRIC
        else:
            exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'

    exp_log.to_csv(exp.log_csv, index=False)

