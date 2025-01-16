import os
import cv2
import numpy as np
from tqdm import tqdm

def undistort_fisheye(rgb_txt, sequence_path, camera_matrix, distortion_coeffs):
    image_list = []
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path, *extra = line.strip().split(' ')
            image_list.append(path)

    first = True
    for image_name in tqdm(image_list):
        image_path = os.path.join(sequence_path, image_name)
        image = cv2.imread(image_path)
        if first:
            first = False
            h, w = image.shape[:2]
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix, distortion_coeffs, (w, h), None)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(image_path, undistorted_image)

    fx, fy, cx, cy = (new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                      new_camera_matrix[0, 2], new_camera_matrix[1, 2])
    return fx, fy, cx, cy
