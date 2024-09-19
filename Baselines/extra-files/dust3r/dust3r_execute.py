from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from downsample_rgb_frames import downsample_rgb_frames

import os
import torch
import subprocess
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument('sequence_path', type=str, help="Path to the sequence directory.")
    parser.add_argument('exp_folder', type=str, help=f"Path to sequence.")
    parser.add_argument('exp_id', type=str, help=f"Path to sequence.")
    parser.add_argument('--max_rgb', type=int, default=10, help="Maximum number of RGB images.")
    parser.add_argument('--min_fps', type=float, default=10.0, help="Minimum downsampled frames per second.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")

    parser.add_argument('--device', type=str, default='cuda', help="")
    parser.add_argument('--batch_size', type=int, default=1, help="")
    parser.add_argument('--schedule', type=str, default='cosine', help="")
    parser.add_argument('--lr', type=float, default=0.01, help="")
    parser.add_argument('--niter', type=int, default=300, help="")
    parser.add_argument('--img_size', type=int, default=512, help="")

    args = parser.parse_args()
    sequence_path = args.sequence_path.replace("sequence_path:", "")
    rgb_txt = os.path.join(sequence_path, 'rgb.txt')
    exp_folder = args.exp_folder.replace('exp_folder:', '')
    exp_id = args.exp_id.replace('exp_id:', '')

    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    img_size = args.img_size

    # Verify if PyTorch is compiled with CUDA
    print("\nVerify if PyTorch is compiled with CUDA: ")
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print(output)
    except FileNotFoundError:
        print("    nvcc (NVIDIA CUDA Compiler) not found, CUDA may not be installed.")

    print(f"    Torch with CUDA is available: {torch.cuda.is_available()}")
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    Device capability: {torch.cuda.get_device_capability(0)}")

    downsampled_paths, downsampled_timestamps = downsample_rgb_frames(rgb_txt, args.max_rgb, args.min_fps)
    for i_path, downsampled_path in enumerate(downsampled_paths):
        downsampled_paths[i_path] = os.path.join(sequence_path, downsampled_path)


    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    images = load_images(downsampled_paths, size=img_size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    #view1, pred1 = output['view1'], output['pred1']
    #view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    poses = scene.get_im_poses()
    keyFrameTrajectory_txt = os.path.join(exp_folder, exp_id.zfill(5) + '_KeyFrameTrajectory' + '.txt')
    with open(keyFrameTrajectory_txt, 'w') as file:
        for i_pose, pose in enumerate(poses):
            tx, ty, tz = pose[0, 3].item(), pose[1, 3].item(), pose[2, 3].item()
            rotation_matrix = np.array([[pose[0, 0].item(), pose[0, 1].item(), pose[0, 2].item()],
                                        [pose[1, 0].item(), pose[1, 1].item(), pose[1, 2].item()],
                                        [pose[2, 0].item(), pose[2, 1].item(), pose[2, 2].item()]])
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            ts = downsampled_timestamps[i_pose]
            line = str(ts) + " " + str(tx) + " " + str(ty) + " " + str(tz) + " " + str(qx) + " " + str(
                qy) + " " + str(qz) + " " + str(qw) + "\n"
            file.write(line)

    # visualize reconstruction
    if args.verbose:
        scene.show()
