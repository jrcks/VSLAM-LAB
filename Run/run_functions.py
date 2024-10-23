# Run methods

import time
import os
import shutil

from Baselines.baseline_utilities import log_run_sequence_time
from path_constants import RGB_BASE_FOLDER
from Run import ablations
from Baselines.downsample_rgb_frames import downsample_rgb_frames

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def run_sequence(exp_it, exp, baseline, dataset, sequence_name, ablation=False):
    print(f"\n{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")
    run_time_start = time.time()

    # Create experiment folder
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)

    # Select images
    create_rgb_exp_txt(exp, dataset, sequence_name)

    # Build execution command
    exec_command = baseline.build_execute_command(exp_it, exp, dataset, sequence_name)

    # Prepare Ablation
    if ablation:
        exec_command = ablations.prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command)

    # Execute experiment
    baseline.execute(exec_command, exp_it, exp_folder)

    # Finish Ablation
    if ablation:
        ablations.finish_ablation(exp_it, baseline, dataset, sequence_name)

    # Log iteration duration
    duration_time = time.time() - run_time_start
    log_run_sequence_time(exp_folder, exp_it, duration_time)

    return duration_time

def create_rgb_exp_txt(exp, dataset, sequence_name):
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    rgb_txt = os.path.join(sequence_path, f"{RGB_BASE_FOLDER}.txt")
    rgb_exp_txt = os.path.join(exp_folder, f"{RGB_BASE_FOLDER}_exp.txt")

    if not os.path.exists(rgb_exp_txt):
        shutil.copy(rgb_txt, rgb_exp_txt)

    if 'max_rgb' in exp.parameters:
        min_fps = dataset.rgb_hz / 10
        downsampled_paths, downsampled_timestamps = downsample_rgb_frames(rgb_txt, exp.parameters['max_rgb'], min_fps, True)

        with open(rgb_exp_txt, 'w') as file:
            for timestamp, path in zip(downsampled_timestamps, downsampled_paths):
                file.write(f"{timestamp} {path}\n")
