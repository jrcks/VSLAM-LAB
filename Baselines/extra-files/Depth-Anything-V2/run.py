import argparse
import cv2
import numpy as np
import os
import torch
import json
from tqdm import tqdm
import sys
import shutil

Depth_Anything_V2_path = os.path.join(os.getcwd(), 'Baselines', 'Depth-Anything-V2')
checkpoints_path = os.path.join(Depth_Anything_V2_path, 'checkpoints')
sys.path.append(Depth_Anything_V2_path)

from depth_anything_v2.dpt import DepthAnythingV2

def analyze_image(raw_image):
    if raw_image is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")

    # Flatten the image to 1D for statistical analysis
    flat_image = raw_image.flatten()

    # Calculate statistics
    mean_val = np.mean(flat_image)
    median_val = np.median(flat_image)
    max_val = np.max(flat_image)
    min_val = np.min(flat_image)

    # Find the minimum value different from zero
    non_zero_pixels = flat_image[flat_image > 0]  # Exclude zero values
    min_non_zero_val = np.min(non_zero_pixels) if non_zero_pixels.size > 0 else 0

    stats = {
        "mean": mean_val,
        "median": median_val,
        "max": max_val,
        "min": min_val,
        "min_non_zero": min_non_zero_val,
    }
    print("Image Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--sequence_path', type=str)
    parser.add_argument('--rgb_txt', type=str)

    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--max-depth', type=float, default=13)

    args = parser.parse_args()
    sequence_path = args.sequence_path
    rgb_txt = args.rgb_txt

    # Check depth folder
    depth_folder = os.path.join(sequence_path, 'depth_anything_v2')
    #if os.path.exists(depth_folder):
    #    shutil.rmtree(depth_folder)
    os.makedirs(depth_folder, exist_ok=True)

    rgbd_txt = os.path.join(sequence_path, 'rgbd_depth_anything_v2.txt')
    if os.path.isfile(rgbd_txt):
        os.remove(rgbd_txt)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder]})
    depth_anything.load_state_dict(torch.load(os.path.join(checkpoints_path, f'depth_anything_v2_{args.encoder}.pth'), map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Load images from rgb.txt
    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            rgb_paths.append(os.path.join(sequence_path, parts[1]))
            rgb_timestamps.append(parts[0])

    rgbd_assoc = []
    for k, filename in enumerate(tqdm(rgb_paths)):

        # raw_image = cv2.imread(filename)
        # depth = depth_anything.infer_image(raw_image, args.input_size)
        #
        # zero_values = depth < 0.00000001
        # depth[zero_values] = -1.0
        # depth = (1.0 / depth)
        # depth[zero_values] = 0.0
        #
        rgbImage = filename
        depthImage = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.png')
        # depthImage = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.npy')
        # jsonFile = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.json')
        #
        # metadata = {
        #     "depthMin": str(depth.min()),
        #     "depthMax": str(depth.max()),
        #     "normalizationFactor": str(65535),
        #     "rgbImage": str(rgbImage),
        #     "depthImage": str(depthImage),
        #     "normalization": str("depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535"),
        #     "type": str("uint16"),
        #     "Description": "Normalized Depth image"
        # }
        #
        # np.save(depthImage, depth)

        # with open(jsonFile, 'w') as json_file:
        #     json.dump(metadata, json_file)

        rgbd_assoc.append(f"{rgb_timestamps[k]} rgb/{os.path.basename(rgbImage)}  "
                          f"{rgb_timestamps[k]} depth_anything_v2/{os.path.basename(depthImage)}")

    with open(rgbd_txt, 'w') as file:
        for line in rgbd_assoc:
            file.write(line + '\n')
