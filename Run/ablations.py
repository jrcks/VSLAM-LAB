import os
import cv2
import shutil
import numpy as np
import yaml
import inspect
import pandas as pd

from path_constants import RGB_BASE_FOLDER
from path_constants import ABLATION_PARAMETERS_CSV
from utilities import ws
from Baselines.baseline_utilities import append_ablation_parameters_to_csv

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command):
    print(f"{ws(8)}Sequence {dataset.dataset_color}{sequence_name}\033[0m preparing ablation ...")
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    settings_yaml = baseline.settings_yaml

    # Start log for ablation parameters
    ablation_parameters = {}
    ablation_parameters['expId'] = str(exp_it).zfill(5)
    ablation_parameters_csv = os.path.join(exp_folder, ABLATION_PARAMETERS_CSV)

    ## Define ablations

    # Ablation: add noise to images
    if 'image_noise' in exp.parameters:
        ablation_parameters_i = add_noise_to_images_start(exp_it, exp, dataset, sequence_name)
        ablation_parameters.update(ablation_parameters_i)

    # Ablation: modify parameters
    if 'ablation_param' in exp.parameters:
        settings_ablation_yaml = settings_yaml.replace('_settings', '_settings_ablation')
        if os.path.exists(settings_ablation_yaml):
            os.remove(settings_ablation_yaml)
        shutil.copy(settings_yaml, settings_ablation_yaml)

        ablation_params = exp.parameters['ablation_param']
        for ablation_param in ablation_params:
            ablation_parameters_i = parameter_ablation_start(exp_it, ablation_param, settings_ablation_yaml)
            ablation_parameters.update(ablation_parameters_i)

        append_ablation_parameters_to_csv(ablation_parameters_csv, ablation_parameters)
        exec_command = exec_command.replace(settings_yaml, settings_ablation_yaml)

    return exec_command


def finish_ablation(exp_it, baseline, dataset, sequence_name):
    print(f"{ws(8)}Sequence '{sequence_name}' finishing ablation ...")
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    add_noise_to_images_finish(sequence_path, exp_it)
    parameter_ablation_finish(baseline)


def modify_yaml_parameter(yaml_file, section_name, parameter_name, new_value):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    if section_name in data and parameter_name in data[section_name]:
        data[section_name][parameter_name] = new_value
        print(f"    Parameter '{parameter_name}' in section '{section_name}' updated to '{new_value}'.")
    else:
        print(f"    Parameter '{parameter_name}' or section '{section_name}' not found in the YAML file.")

    # Write the changes back to the YAML file
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)

    print(f"    YAML file '{yaml_file}' has been updated.")


def parameter_ablation_start(exp_it, ablation_param, settings_ablation_yaml):
    min_exp = -5
    max_exp = 2
    num_it = 100
    b = min_exp
    m = (max_exp - min_exp) / (num_it - 1)

    def parameter_ablation(it_):
        it__ = (it_ % num_it)
        return 10 ** (m * it__ + b)

    source_code = inspect.getsource(parameter_ablation)
    parameter_policy = source_code[source_code.find('return') + len('return'):].strip()

    print(f"{SCRIPT_LABEL} Parameter policy: {ablation_param} = {parameter_policy}")
    value = parameter_ablation(exp_it)
    print(f"    it = {exp_it}")
    print(f"    ablation value = {value}")

    section_name, parameter_name = ablation_param.split('.', 1)
    modify_yaml_parameter(settings_ablation_yaml, section_name, parameter_name, value)

    ablation_parameters = {ablation_param: value}

    return ablation_parameters


def parameter_ablation_finish(baseline):
    settings_yaml = baseline.settings_yaml
    settings_ablation_yaml = settings_yaml.replace('_settings', '_settings_ablation')
    if os.path.exists(settings_ablation_yaml):
        os.remove(settings_ablation_yaml)


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


def find_groundtruth_txt(trajectories_path, trajectory_file, exp):
    parameter = exp.parameters['ablation_param'][0]
    ablation_parameters_csv = os.path.join(trajectories_path, ABLATION_PARAMETERS_CSV)
    traj_name = os.path.basename(trajectory_file)
    df = pd.read_csv(ablation_parameters_csv)
    index_str = traj_name.split('_')[0]
    expId = int(index_str)
    exp_row = df[df['expId'] == expId]
    ablation_values = exp_row[parameter].values[0]

    df_noise_filter = df[df['std_noise'] == 0]
    gt_id = df_noise_filter[(df_noise_filter[parameter].sub(ablation_values).abs() == df_noise_filter[parameter].sub(
        ablation_values).abs().min())]
    if gt_id.loc[gt_id['expId'] == expId].empty:
        gt_id = np.random.choice(gt_id['expId'].values)
    else:
        gt_id = expId

    groundtruth_txt = os.path.join(trajectories_path, f"{str(gt_id).zfill(5)}_KeyFrameTrajectory.txt")
    return groundtruth_txt
