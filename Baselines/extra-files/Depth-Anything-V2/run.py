import argparse
import cv2
import numpy as np
import os
import torch
import json
from tqdm import tqdm
import sys

Depth_Anything_V2_path = os.path.join(os.getcwd(), 'Baselines', 'Depth-Anything-V2', 'metric_depth')
checkpoints_path = os.path.join(Depth_Anything_V2_path, 'checkpoints')
sys.path.append(Depth_Anything_V2_path)

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

    parser.add_argument('--sequence_path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default=os.path.join(checkpoints_path, 'depth_anything_v2_metric_hypersim_vitl.pth'))
    parser.add_argument('--max-depth', type=float, default=20)

    args = parser.parse_args()
    sequence_path = args.sequence_path

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Find rgb subfolders
    rgb_txt_files = []
    for item in os.listdir(sequence_path):
        item_path = os.path.join(sequence_path, item)
        if os.path.isfile(item_path) and 'rgb' in item.lower():
            rgb_txt_files.append(item_path)

    rgb_paths = {}
    rgb_timestamps = {}

    for rgb_txt in rgb_txt_files:
        rgb_paths[rgb_txt] = []
        rgb_timestamps[rgb_txt] = []

        with open(rgb_txt, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                rgb_paths[rgb_txt].append(os.path.join(sequence_path, parts[1]))
                rgb_timestamps[rgb_txt].append(parts[0])


    for filenames in rgb_paths.values():
        outdir = os.path.join(sequence_path, 'depth')
        os.makedirs(outdir, exist_ok=True)
        rgbd_assoc_txt = os.path.join(sequence_path, 'rgbd_assoc.txt')
        rgbd_assoc = []

        for k, filename in enumerate(tqdm(filenames)):
            raw_image = cv2.imread(filename)

            depth = depth_anything.infer_image(raw_image, args.input_size)

            rgbImage = filename
            depthImage = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            jsonFile = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.json')

            metadata = {
                "depthMin": str(depth.min()),
                "depthMax": str(depth.max()),
                "rgbImage": str(rgbImage),
                "depthImage": str(depthImage),
                "normalization": str("depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0"),
                "type": str("uint8"),
                "Description": "Normalized Depth image"
            }

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            cv2.imwrite(depthImage, depth)

            with open(jsonFile, 'w') as json_file:
                json.dump(metadata, json_file)

            rgbd_assoc.append(f"{rgb_timestamps[rgb_txt][k]} rgb/{os.path.basename(rgbImage)}  "
                              f"depth/{os.path.basename(depthImage)} "
                              f"depth/{os.path.basename(jsonFile)}")

        with open(rgbd_assoc_txt, 'w') as file:
            for line in rgbd_assoc:
                file.write(line + '\n')