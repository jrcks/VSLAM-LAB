import os
import yaml
import re
import pandas as pd

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import decompressFile

from Datasets.download_utilities import download_file_from_google_drive
from Datasets.download_utilities import create_google_drive_service

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics

from utilities import ws
from utilities import VSLAM_LAB_DIR


class DRUNKARDS_dataset(DatasetVSLAMLab):

    google_drive_service = 0

    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('drunkards', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']
        self.client_secrets_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', data['client_secrets_file'])

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Ground-Truth ids
        self.rgb_ids = data['rgb_ids']

        # Ground-Truth ids
        self.ground_truth_ids = data['ground_truth_ids']

        # Intrinsics_id
        self.intrinsics_ids = data['intrinsics_ids']

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Download rgb images
        rgb_id = self.rgb_ids[sequence_name]
        compressed_name_ext = sequence_name + '.zip'
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, sequence_name)
        download_file_from_google_drive(rgb_id, compressed_file, DRUNKARDS_dataset.google_drive_service)
        decompressFile(compressed_file, decompressed_folder)

        # Download ground truth
        groundtruth_id = self.ground_truth_ids[sequence_name]
        groundtruth_txt = os.path.join(sequence_path, 'pose.txt')
        download_file_from_google_drive(groundtruth_id, groundtruth_txt, DRUNKARDS_dataset.google_drive_service)

        # Download intrinsics
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_id = self.intrinsics_ids[resolution]
        intrinsics_txt = os.path.join(sequence_path, f'intrinsics_{resolution}.txt')
        download_file_from_google_drive(intrinsics_id, intrinsics_txt, DRUNKARDS_dataset.google_drive_service)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        color_path = os.path.join(sequence_path, 'color')
        rgb_path = os.path.join(sequence_path, 'rgb')
        if os.path.exists(color_path) and os.path.isdir(color_path):
            os.rename(color_path, rgb_path)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                file.write(f"{ts:.16f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_txt = os.path.join(sequence_path, f'intrinsics_{resolution}.txt')
        with open(intrinsics_txt, 'r') as file:
            for line in file:
                if line.startswith('fx, fy, cx, cy:'):
                    values_str = line.split(':')[1].strip()
                    fx, fy, cx, cy = map(float, values_str.split(','))

        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml(fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        pose_txt = os.path.join(sequence_path, 'pose.txt')
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        if os.path.exists(pose_txt):
            os.rename(pose_txt, groundtruth_txt)
            data = pd.read_csv(groundtruth_txt, sep=' ', header=None)
            data[0] = data[0] / self.rgb_hz
            data.to_csv(groundtruth_txt, sep=' ', header=False, index=False, float_format='%.16f')

    def remove_unused_files(self, sequence_name):
        compressed_file = os.path.join(self.dataset_path, sequence_name + '.zip')
        os.remove(compressed_file)

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_txt = os.path.join(sequence_path, f'intrinsics_{resolution}.txt')
        os.remove(intrinsics_txt)

    def evaluate_trajectory_accuracy(self, trajectory_txt, groundtruth_txt):
        traj_xyz_aligned, gt_xyz, traj_xyz_full_aligned, gt_xyz_full = align_trajectory_with_groundtruth(
            trajectory_txt, groundtruth_txt, 1.0 / self.rgb_hz, 1.0, 0)

        rmse_ate = metrics.rmse_ate(traj_xyz_aligned, gt_xyz)
        return rmse_ate, len(traj_xyz_aligned), traj_xyz_aligned, gt_xyz, gt_xyz_full

    def get_sequence_resolution(self, sequence_name):
        return int(re.search(r'_(\d+)_', sequence_name).group(1))

    def get_download_issues(self, sequence_name):
        issues = {'Google Drive': f"Downloading this dataset requires to grant access to Google Drive."}

        return issues

    def solve_download_issue(self, download_issue):
        if download_issue[0] == 'Google Drive':
            print(f"{ws(4)}[{self.dataset_name}][{download_issue[0]}]: {download_issue[1]} ")
            DRUNKARDS_dataset.google_drive_service = create_google_drive_service(self.url_download_root,
                                                                                 self.client_secrets_file)
            print(f"{ws(4)}[{self.dataset_name}][{download_issue[0]}]: Access to Google Drive granted. ")
