import subprocess
import os

from Evaluate.evo_functions import evo_get_accuracy
from path_constants import VSLAM_LAB_EVALUATION_FOLDER

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def evaluate_sequence(exp, dataset, sequence_name, ablation=None):
    trajectories_path = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    groundtruth_file = os.path.join(sequence_path, 'groundtruth.txt')

    # Groundtruth evaluation
    evaluation_folder = os.path.join(trajectories_path, VSLAM_LAB_EVALUATION_FOLDER)
    pseudo_groundtruth = 0
    os.makedirs(evaluation_folder, exist_ok=True)

    print(f"\n{SCRIPT_LABEL}Evaluating '{evaluation_folder}' in {dataset.dataset_color}{sequence_name}\033[0m:")

    print(f"\n{SCRIPT_LABEL}ATE evaluation: {groundtruth_file}")
    command = (f"pixi run -e evo python Evaluate/evo_functions.py ate {1.0 / dataset.rgb_hz} {trajectories_path} {evaluation_folder} {groundtruth_file} {pseudo_groundtruth}")
    subprocess.run(command, shell=True)

    #print(f"\n{SCRIPT_LABEL}RPE comparison: {groundtruth_file}")
    #command = (f"pixi run -e evo python Evaluate/evo_functions.py rpe {1.0 / dataset.rgb_hz} {trajectories_path} {evaluation_folder} {groundtruth_file} {pseudo_groundtruth}")
    #subprocess.run(command, shell=True)

    # Pseudo evaluation
    # if ablation:
    #     evaluation_folder = os.path.join(trajectories_path, f"{VSLAM_LAB_EVALUATION_FOLDER}_pseudo")
    #     pseudo_groundtruth = 1
    #     os.makedirs(evaluation_folder, exist_ok=True)
    #
    #     print(f"\n{SCRIPT_LABEL}Evaluating '{evaluation_folder}' in {dataset.dataset_color}{sequence_name}\033[0m:")
    #
    #     print(f"\n{SCRIPT_LABEL}ATE pseudo evaluation")
    #     command = (f"pixi run -e evo python Evaluate/evo_functions.py ate {1.0 / dataset.rgb_hz} {trajectories_path} {evaluation_folder} {groundtruth_file} {pseudo_groundtruth} {exp.parameters['ablation_param'][0]}")
    #     subprocess.run(command, shell=True)

        #print(f"\n{SCRIPT_LABEL}RPE pseudo comparison")
        #command = (f"pixi run -e evo python Evaluate/evo_functions.py rpe {1.0 / dataset.rgb_hz} {trajectories_path} {evaluation_folder} {groundtruth_file} {pseudo_groundtruth} {exp.parameters['ablation_param'][0]}")
        #subprocess.run(command, shell=True)

