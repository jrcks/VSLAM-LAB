import os
import cv2
import shutil
import numpy as np
import yaml

from utilities import VSLAM_LAB_BASELINES_DIR


def modify_yaml_parameter(yaml_file, section_name, parameter_name, new_value):

    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    if section_name in data and parameter_name in data[section_name]:
        data[section_name][parameter_name] = new_value
        print(f"Parameter '{parameter_name}' in section '{section_name}' updated to '{new_value}'.")
    else:
        print(f"Parameter '{parameter_name}' or section '{section_name}' not found in the YAML file.")

    # Write the changes back to the YAML file
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)

    print(f"YAML file '{yaml_file}' has been updated.")

def glomap_parameter_ablation_start(it):
    glomap_dir = os.path.join(VSLAM_LAB_BASELINES_DIR, 'glomap')
    glomap_settings_yaml = os.path.join(glomap_dir, 'glomap_settings.yaml')
    glomap_settings_saved_yaml = os.path.join(glomap_dir, 'glomap_settings_saved.yaml')

    if os.path.exists(glomap_settings_saved_yaml):
        shutil.copy(glomap_settings_saved_yaml, glomap_settings_yaml)
    else:
        shutil.copy(glomap_settings_yaml, glomap_settings_saved_yaml)

    value = 0.1 + it * 5
    modify_yaml_parameter(glomap_settings_yaml, 'BundleAdjustment', 'thres_loss_function', value)

def glomap_parameter_ablation_finish():
    glomap_dir = os.path.join(VSLAM_LAB_BASELINES_DIR, 'glomap')
    glomap_settings_yaml = os.path.join(glomap_dir, 'glomap_settings.yaml')
    glomap_settings_saved_yaml = os.path.join(glomap_dir, 'glomap_settings_saved.yaml')

    shutil.copy(glomap_settings_saved_yaml, glomap_settings_yaml)
    os.remove(glomap_settings_saved_yaml)

def add_noise_to_images_start(sequence_path, it):

    # Save rgb folder
    rgb_path = os.path.join(sequence_path, 'rgb')
    rgb_path_saved = os.path.join(sequence_path, 'rgb_saved')
    if not os.path.exists(rgb_path_saved):
        os.rename(rgb_path, rgb_path_saved)
    os.makedirs(os.path.join(sequence_path, 'rgb'), exist_ok=True)

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

    std_noise = it * 1.0
    print(it)
    print(std_noise)

    def add_gaussian_noise(image_, mean=0, std_dev=25):
        noise = np.random.normal(mean, std_dev, image_.shape).astype(np.float32)
        noisy_image_ = image_ + noise
        noisy_image_ = np.clip(noisy_image_, 0, 255).astype(np.uint8)
        return noisy_image_

    for i, rgb_file_saved in enumerate(rgb_files_saved):
        rgb_file = rgb_files[i]
        image = cv2.imread(os.path.join(sequence_path, rgb_file_saved))
        noisy_image = add_gaussian_noise(image, mean=0, std_dev= std_noise)
        cv2.imwrite(os.path.join(sequence_path, rgb_file), noisy_image)

def add_noise_to_images_finish(sequence_path):

    # Remove rgb_saved.txt
    rgb_txt_saved = os.path.join(sequence_path, 'rgb_saved.txt')
    os.remove(rgb_txt_saved)

    # Restore rgb folder
    rgb_path = os.path.join(sequence_path, 'rgb')
    rgb_path_saved = os.path.join(sequence_path, 'rgb_saved')
    shutil.rmtree(rgb_path)
    os.rename(rgb_path_saved, rgb_path)
