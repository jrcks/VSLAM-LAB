"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-12
- Updated: 2024-07-12
- License: GPLv3 License
- List of Known Dependencies;
    * ...

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.
Specifically downloading sequences, running experiments, and evaluating results.

"""

import csv
import glob
import os
import shutil
import subprocess
import sys

import cv2
import pandas as pd
import yaml
import numpy as np

from utilities import VSLAM_LAB_DIR
from utilities import find_files_with_string
from utilities import ws
from utilities import check_sequence_integrity

SCRIPT_LABEL = f"[{os.path.basename(__file__)}] "


class DatasetVSLAMLab:

    def __init__(self, dataset_name, benchmark_path):

        # Init dataset paths
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_name.upper()
        self.benchmark_path = benchmark_path
        self.dataset_path = os.path.join(self.benchmark_path, self.dataset_folder)

        self.yaml_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_' + self.dataset_name + '.yaml')

        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        self.sequence_names = data['sequence_names']
        self.rgb_hz = data['rgb_hz']
        self.sequence_nicknames = []

    ####################################################################################################################
    # Download methods
    def download_sequence(self, sequence_name):

        # Check if sequence is already available
        sequence_availability = self.check_sequence_availability(sequence_name)
        if sequence_availability == "available":
            print(f"{ws(4)}Sequence '{sequence_name}' is already downloaded.")
            return
        if sequence_availability == "corrupted":
            print(f"{ws(8)}Some files in sequence {sequence_name} are corrupted.")
            print(f"{ws(8)}Removing and downloading again sequence {sequence_name} ")
            print(f"{ws(8)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY")
            sys.exit(1)

        # Download process
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)

        self.download_process(sequence_name)

    def download_process(self, sequence_name):
        msg = f"Downloading sequence '{sequence_name}' from dataset '{self.dataset_name}' ..."
        print(SCRIPT_LABEL + msg)
        self.download_sequence_data(sequence_name)
        self.create_rgb_folder(sequence_name)
        self.create_rgb_txt(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_txt(sequence_name)
        self.remove_unused_files(sequence_name)

    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        return

    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_txt(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        return

    def get_download_issues(self, sequence_name):
        return {}

    def solve_download_issue(self, download_issue):
        return

    def write_calibration_yaml(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name):

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        image_0 = cv2.imread(os.path.join(rgb_path, rgb_files[0]))
        h, w, channels = image_0.shape

        yaml_content_lines = [
            "%YAML:1.0",
            "",
            "# Camera calibration and distortion parameters (OpenCV)",
            "Camera.fx: " + str(fx),
            "Camera.fy: " + str(fy),
            "Camera.cx: " + str(cx),
            "Camera.cy: " + str(cy),
            "",
            "Camera.k1: " + str(k1),
            "Camera.k2: " + str(k2),
            "Camera.p1: " + str(p1),
            "Camera.p2: " + str(p2),
            "Camera.k3: " + str(k3),
            "",
            "Camera.w: " + str(w),
            "Camera.h: " + str(h),
            "",
            "# Camera frames per second",
            "Camera.fps: " + str(self.rgb_hz)
        ]

        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def check_sequence_availability(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if os.path.exists(sequence_path):
            sequence_complete = check_sequence_integrity(self.dataset_path, sequence_name, True)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

    ####################################################################################################################
    # Run methods

    def run_sequence(self, exp, sequence_name_, ablation=False):
        sequence_name = sequence_name_

        sequence_path = os.path.join(self.dataset_path, sequence_name)

        output_path = os.path.join(exp.folder, self.dataset_folder, sequence_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        search_pattern = os.path.join(output_path, f'*system_output_*')
        system_output_files = glob.glob(search_pattern)
        it = len(system_output_files)
        log_file_path = os.path.join(output_path, "system_output_" + str(it).zfill(5) + ".txt")

        exec_command = [f"sequence_path:{sequence_path}", f"exp_folder:{output_path}", f"exp_id:{it}"]
        i_par = 0
        for parameter in exp.parameters:
            exec_command += [str(parameter)]
            i_par += 1

        command_str = ' '.join(exec_command)

        full_command = f"pixi run -e {exp.vslam} execute " + command_str

        if ablation:
            self.prepare_ablation(sequence_name, exp, it)
        self.run_executable(full_command, log_file_path)
        if ablation:
            self.finish_ablation(sequence_name)

    def run_executable(self, command, log_file_path):
        with open(log_file_path, 'w') as log_file:
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(command, stdout=log_file, stderr=log_file, shell=True)

    ####################################################################################################################

    def add_gaussian_noise(self, image, mean=0, std_dev=25):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    # Ablation methods
    def prepare_ablation(self, sequence_name, exp, it):
        print(f"{ws(8)}Sequence '{sequence_name}' preparing ablation ...")
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Save rgb folder
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_path_saved = os.path.join(sequence_path, 'rgb_saved')
        if not os.path.exists(rgb_path_saved):
            os.rename(rgb_path, rgb_path_saved)
        os.makedirs(os.path.join(sequence_path,'rgb'), exist_ok=True)

        # update rgb.txt
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        rgb_txt_saved = os.path.join(sequence_path, 'rgb_saved.txt')
        with open(rgb_txt, 'r') as file:
             content = file.read()
        modified_content = content.replace('rgb', 'rgb_saved')
        with open(rgb_txt_saved, 'w') as file:
            file.write(modified_content)

        # update rgb folder
        rgb_files_saved = []
        with open(rgb_txt_saved, 'r') as file:
            for line in file:
                timestamp, path = line.strip().split(' ')
                rgb_files_saved.append(path)

        rgb_files = []
        with open(rgb_txt, 'r') as file:
            for line in file:
                timestamp, path = line.strip().split(' ')
                rgb_files.append(path)

        std_noise = (it) * 0.25
        print(it)
        print(std_noise)
        for i, rgb_file_saved in enumerate(rgb_files_saved):
            rgb_file = rgb_files[i]
            image = cv2.imread(os.path.join(sequence_path, rgb_file_saved))
            noisy_image = self.add_gaussian_noise(image, mean=0, std_dev= std_noise)
            cv2.imwrite(os.path.join(sequence_path, rgb_file), noisy_image)

    def finish_ablation(self, sequence_name):
        print(f"{ws(8)}Sequence '{sequence_name}' finishing ablation ...")
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Remove rgb_saved.txt
        rgb_txt_saved = os.path.join(sequence_path, 'rgb_saved.txt')
        os.remove(rgb_txt_saved)

        # Restore rgb folder
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_path_saved = os.path.join(sequence_path, 'rgb_saved')
        shutil.rmtree(rgb_path)
        os.rename(rgb_path_saved, rgb_path)

    ####################################################################################################################
    # Evaluation methods
    def evaluate_sequence(self, sequence_name, experiment_folder):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_file = os.path.join(sequence_path, 'groundtruth.txt')

        trajectories_path = os.path.join(experiment_folder, self.dataset_folder, sequence_name)
        evaluation_folder = os.path.join(trajectories_path, 'vslamlab_evaluation')
        if os.path.exists(evaluation_folder):
            shutil.rmtree(evaluation_folder)
        os.makedirs(evaluation_folder, exist_ok=True)

        trajectory_files = find_files_with_string(trajectories_path, "_KeyFrameTrajectory.txt")

        print(f"{ws(4)}Evaluation of '{os.path.basename(experiment_folder)}"
              f"' in '{sequence_name}': {len(trajectory_files)} trajectories")

        # Compute accuracy
        all_traj_accuracies = []
        all_num_evaluation_pts = []
        for iTraj, trajectory_i in enumerate(trajectory_files):
            traj_accuracy, num_eval_pts, traj_xyz_aligned, gt_xyz, gt_xyz_full \
                = self.evaluate_trajectory_accuracy(trajectory_i, groundtruth_file)
            print(trajectory_i)
            print(traj_accuracy)
            all_traj_accuracies.append(traj_accuracy)
            all_num_evaluation_pts.append(num_eval_pts)

            df_traj = pd.DataFrame(traj_xyz_aligned, columns=['tx', 'ty', 'tz'])
            df_gt = pd.DataFrame(gt_xyz, columns=['tx gt', 'ty gt', 'tz gt'])
            df_traj_gt = pd.concat([df_traj, df_gt], axis=1)

            trajectory_csv = os.path.join(evaluation_folder, f'aligned_traj_{os.path.basename(trajectory_i)}.csv')
            trajectory_csv = trajectory_csv.replace("_KeyFrameTrajectory.txt", '')

            with open(trajectory_csv, 'w', newline='') as file:
                df_traj_gt.to_csv(file, index=False)

        df_gt_full = pd.DataFrame(gt_xyz_full, columns=['tx gt', 'ty gt', 'tz gt'])
        gt_csv = os.path.join(evaluation_folder, f'gt.csv')
        with open(gt_csv, 'w', newline='') as file:
            df_gt_full.to_csv(file, index=False)

        accuracy_csv = os.path.join(evaluation_folder, 'accuracy.csv')
        with open(accuracy_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trajectory Accuracy', 'Number of Evaluation Points'])
            writer.writerows(zip(all_traj_accuracies, all_num_evaluation_pts))

    ####################################################################################################################
    # Utils

    def contains_sequence(self, sequence_name_ref):
        for sequence_name in self.sequence_names:
            if sequence_name == sequence_name_ref:
                return True
        return False

    def print_sequence_names(self):
        print(self.sequence_names)

    def print_sequence_nicknames(self):
        print(self.sequence_nicknames)

    def get_sequence_names(self):
        return self.sequence_names

    def get_sequence_nicknames(self):
        return self.sequence_nicknames

    def get_sequence_nickname(self, sequence_name_ref):
        for i, sequence_name in enumerate(self.sequence_names):
            if sequence_name == sequence_name_ref:
                return self.sequence_nicknames[i]

    def get_sequence_num_rgb(self, sequence_name):
        rgb_txt = os.path.join(self.dataset_path, sequence_name, 'rgb.txt')
        if os.path.exists(rgb_txt):
            with open(rgb_txt, 'r') as file:
                line_count = 0
                for line in file:
                    line_count += 1
            return line_count
        return 0
