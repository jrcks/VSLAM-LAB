import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

def get_colmap_keyframes(images_file, number_of_header_lines, verbose=False):
    print(f"get_colmap_keyframes: {images_file}")
    
    image_id = []
    q_wc_xyzw = []
    t_wc = []

    with open(f"{images_file}", 'r') as file:
        # Skip the header lines
        for _ in range(number_of_header_lines):
            file.readline()
        
        while True:
            line1 = file.readline()
            if not line1:
                break
            elements = line1.split()
            
            IMAGE_ID = int(elements[0])
            image_id.append(IMAGE_ID)
            
            QW = float(elements[1])
            QX = float(elements[2])
            QY = float(elements[3])
            QZ = float(elements[4])
            
            TX = float(elements[5])
            TY = float(elements[6])
            TZ = float(elements[7])

            t_cw_i = np.array([TX, TY, TZ])
            q_wc_i = R.from_quat([QX, QY, QZ, QW]).inv()
            R_wc_i = q_wc_i.as_matrix()
            
            q_wc_xyzw.append([q_wc_i.as_quat()[1], q_wc_i.as_quat()[2], q_wc_i.as_quat()[3], q_wc_i.as_quat()[0]])
            t_wc.append(-R_wc_i @ t_cw_i)

            file.readline()
    
    image_id = np.array(image_id)
    q_wc_xyzw = np.array(q_wc_xyzw)
    t_wc = np.array(t_wc)

    if verbose:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(t_wc[:, 0], t_wc[:, 1], t_wc[:, 2], 'k.')
        ax.set_title("Reconstructed Trajectory")
        plt.show()
    
    return image_id, t_wc, q_wc_xyzw

def write_trajectory_tum_format(file_name, image_ts, t_wc, q_wc_xyzw):
    print(f"writeTrajectoryTUMformat: {file_name}")
    
    data = np.hstack((image_ts.reshape(-1, 1), t_wc, q_wc_xyzw))
    
    with open(file_name, 'w') as file:
        for row in data:
            # Format the row with the appropriate precision
            file.write(' '.join(f'{x:.15f}' for x in row) + '\n')

def get_timestamps(files_path, rgb_file):
    print(f"getTimestamps: {os.path.join(files_path, rgb_file)}")
    
    ts = []
    
    with open(os.path.join(files_path, rgb_file), 'r') as file:
        for line in file:
            parts = line.split()
            ts.append(float(parts[0]))  # Assuming the first part is the timestamp
    
    return ts
                
if __name__ == "__main__":

    sequence_path = sys.argv[1]
    exp_folder = sys.argv[2]
    exp_id = sys.argv[3]
    images_file = os.path.join(exp_folder, f'colmap_{exp_id}', 'images.txt')

    number_of_header_lines = 4
    image_id, t_wc, q_wc_xyzw = get_colmap_keyframes(images_file, number_of_header_lines)
    
    rgb_file = os.path.join(exp_folder, f'colmap_{exp_id}', 'rgb_ds.txt')
    image_ts = np.array(get_timestamps(sequence_path, rgb_file))
    timestamps = []
    for id in image_id:
        timestamps.append(float(image_ts[id-1]))

    timestamps = np.array(timestamps)

    keyFrameTrajectory_txt = os.path.join(exp_folder, exp_id + '_KeyFrameTrajectory' + '.txt')
    write_trajectory_tum_format(keyFrameTrajectory_txt, timestamps, t_wc, q_wc_xyzw)
