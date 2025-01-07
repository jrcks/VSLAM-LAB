import os
import cv2
import shutil
import numpy as np
import yaml
import inspect
import time
import pandas as pd

from path_constants import RGB_BASE_FOLDER
from path_constants import ABLATION_PARAMETERS_CSV

from utilities import ws
from Baselines.baseline_utilities import append_ablation_parameters_to_csv

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command):
    print(f"\n{SCRIPT_LABEL}Sequence {dataset.dataset_color}{sequence_name}\033[0m preparing ablation: {exp.ablation_csv}")
    #print(f"\n{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")

    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    settings_yaml = baseline.settings_yaml
    ablation_parameters_csv = pd.read_csv(exp.ablation_csv)

    # Create new _settings_ablation.yaml file
    settings_ablation_yaml = os.path.join(exp_folder, os.path.basename(settings_yaml).replace('_settings', '_settings_ablation'))
    if os.path.exists(settings_ablation_yaml):
        os.remove(settings_ablation_yaml)
    shutil.copy(settings_yaml, settings_ablation_yaml)
    exec_command = exec_command.replace(settings_yaml, settings_ablation_yaml)

    # Update _settings_ablation.yaml file
    row = ablation_parameters_csv.iloc[exp_it]
    for parameter, value in row.items():
        if parameter == 'exp_it':
            continue
        if parameter == 'image_noise':
            continue
        section_name, parameter_name = parameter.split('.', 1)
        baseline.modify_yaml_parameter(settings_ablation_yaml, section_name, parameter_name, value)
        print(f"{ws(8)}{section_name}: {parameter_name}: {value}")

    return exec_command


def finish_ablation(exp_it, baseline, dataset, sequence_name):
    print(f"{ws(8)}Sequence '{sequence_name}' finishing ablation ...")
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    #add_noise_to_images_finish(sequence_path, exp_it)



def add_noise_to_images_start(exp_it, exp, dataset, sequence_name):
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    rgb_exp_txt = os.path.join(exp_folder, f"{RGB_BASE_FOLDER}_exp.txt")

    # update rgb.txt
    def std_noise_ablation(it_):
        if it_ < 500:
            return 0.0
        else:
            return exp.parameters['image_noise']

    std_noise = std_noise_ablation(exp_it)
    ablation_parameters = {"std_noise": std_noise}

    source_code = inspect.getsource(std_noise_ablation)
    noise_policy = source_code[source_code.find('return') + len('return'):].strip()
    print(f"{SCRIPT_LABEL} Noise policy: std_noise = {noise_policy}")
    print(f"    exp_it = {exp_it}")
    print(f"    std_noise = {std_noise}")

    if std_noise == 0.0:
        return ablation_parameters

    # Create rgb_path_ablation folder to store new images
    rgb_path_ablation = os.path.join(sequence_path, f"{RGB_BASE_FOLDER}_ablation")
    if os.path.exists(rgb_path_ablation):
        shutil.rmtree(rgb_path_ablation)
        #return ablation_parameters
    os.makedirs(rgb_path_ablation, exist_ok=True)

    def add_gaussian_noise(image_, mean=0, std_dev=25):
        np.random.seed(int(time.time() * 1000) % (2**32))
        noise = np.random.normal(mean, std_dev, image_.shape).astype(np.float32)
        noisy_image_ = image_ + noise
        noisy_image_ = np.clip(noisy_image_, 0, 255).astype(np.uint8)
        return noisy_image_

    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_exp_txt, 'r') as file:
        for line in file:
            timestamp, path = line.strip().split(' ')
            rgb_paths.append(path)
            rgb_timestamps.append(float(timestamp))

    with open(rgb_exp_txt, 'w') as file:
        for i, rgb_path in enumerate(rgb_paths):
            rgb_file = os.path.join(sequence_path, rgb_path)
            rgb_ablation_path = rgb_path.replace(f"{RGB_BASE_FOLDER}/", f"{RGB_BASE_FOLDER}_ablation/")
            rgb_file_ablation = os.path.join(sequence_path, rgb_ablation_path)
            image = cv2.imread(rgb_file)
            noisy_image = add_gaussian_noise(image, mean=0, std_dev=std_noise)
            cv2.imwrite(os.path.join(sequence_path, rgb_file_ablation), noisy_image)
            file.write(f"{rgb_timestamps[i]} {rgb_ablation_path}\n")

    return ablation_parameters


def add_noise_to_images_finish(sequence_path, exp_it):
    #if not (exp_it % 100 == 0):
    #    return

    # Remove rgb folder
    rgb_path_ablation = os.path.join(sequence_path, f"{RGB_BASE_FOLDER}_ablation")
    if os.path.exists(rgb_path_ablation):
        shutil.rmtree(rgb_path_ablation)



