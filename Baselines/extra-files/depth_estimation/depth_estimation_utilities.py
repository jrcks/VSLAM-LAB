import argparse
import shutil
import os
import cv2
import numpy as np
import yaml

def argument_parser(baseline_name):
    parser = argparse.ArgumentParser(description=baseline_name)

    parser.add_argument('--sequence_path', type=str)
    parser.add_argument('--rgb_txt', type=str, default="none", help="Specify the RGB text (default: 'none')")
    parser.add_argument('--max-depth', type=float, default=8.0)
    parser.add_argument('--min-depth', type=float, default=0.5)
    parser.add_argument('--verbose', type=str)

    args, unknown = parser.parse_known_args()
    sequence_path = args.sequence_path

    if args.rgb_txt == 'none':
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
    else:
        rgb_txt = args.rgb_txt

    calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

    depth_folder_name = 'depth_pro'

    max_depth = args.max_depth
    min_depth = args.min_depth

    return sequence_path, rgb_txt, calibration_yaml, max_depth, min_depth, depth_folder_name, bool(int(args.verbose))


def prepare_depth_folder(sequence_path, depth_folder_name):
    depth_folder = os.path.join(sequence_path, depth_folder_name)

    if os.path.exists(depth_folder):
        shutil.rmtree(depth_folder)
    os.makedirs(depth_folder, exist_ok=True)

    rgbd_txt = os.path.join(sequence_path, f'rgbd_{depth_folder_name}.txt')
    if os.path.isfile(rgbd_txt):
        os.remove(rgbd_txt)

    return depth_folder, rgbd_txt


def load_rgb_txt(rgb_txt, sequence_path):
    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            rgb_paths.append(os.path.join(sequence_path, parts[1]))
            rgb_timestamps.append(parts[0])

    return rgb_paths, rgb_timestamps

def save_depth_image(depth_np, depthImage_path, scale_factor,
                     rgbd_assoc, rgb_timestamp, rgbImage_path, depth_folder_name):
    depth_np = scale_factor * depth_np
    depth_np = depth_np.astype(np.uint16)
    cv2.imwrite(depthImage_path, depth_np)

    rgbd_assoc.append(f"{rgb_timestamp} rgb/{os.path.basename(rgbImage_path)} "
                      f"{rgb_timestamp} {depth_folder_name}/{os.path.basename(depthImage_path)}")

    return rgbd_assoc

def load_calibration_yaml(calibration_yaml):

    with open(calibration_yaml, 'r') as file:
        lines = file.readlines()
    if lines and lines[0].strip() == '%YAML:1.0':
        lines = lines[1:]

    calibration = yaml.safe_load(''.join(lines))

    fx, fy, cx, cy = (calibration["Camera.fx"], calibration["Camera.fy"],
                      calibration["Camera.cx"], calibration["Camera.cy"])

    f = 0.5 * (fx + fy)

    depth_factor = calibration["depth_factor"]
    return f, depth_factor

def print_statistics(depth_np):
    min_val = np.min(depth_np)
    max_val = np.max(depth_np)
    median_val = np.median(depth_np)

    print(f"depth_np: {depth_np.shape}")
    print(f"Min Depth Value: {min_val:.4f} m")
    print(f"Max Depth Value: {max_val:.4f} m")
    print(f"Median Depth Value: {median_val:.4f} m")