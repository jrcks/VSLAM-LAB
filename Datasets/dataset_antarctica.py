import os, yaml, shutil
import re
import numpy as np
from PIL import Image
import exifread
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.dataset_utilities import undistort_rgb_rad_tan, resize_rgb_images
from pyproj import CRS, Transformer
from datetime import datetime

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ANTARCTICA_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('antarctica', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.dataset_source_folder = data['dataset_source_folder']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Get resolution size
        self.resolution_size = data['resolution_size']

    # def download_sequence_data(self, sequence_name):
    #     # Variables
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     if not os.path.exists(sequence_path):
    #          os.makedirs(sequence_path)

    #     self.create_groundtruth_txt(sequence_name)
    #     exit(0)

    # def create_rgb_folder(self, sequence_name):
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     rgb_path = os.path.join(sequence_path, 'rgb')
    #     source_rgb_path  = os.path.join(self.dataset_source_folder, sequence_name)
    #     if os.path.exists(rgb_path):
    #         return

    #     os.makedirs(rgb_path, exist_ok=True)     

    #     estimate_new_resolution = True
    #     for file in os.listdir(source_rgb_path):
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
    #             with Image.open(os.path.join(source_rgb_path, file)) as img:
    #                 if estimate_new_resolution:
    #                     scaled_height = np.sqrt(self.resolution_size[0] * self.resolution_size[1] * img.size[1] / img.size[0])
    #                     scaled_width = self.resolution_size[0] * self.resolution_size[1] / scaled_height
    #                     scaled_height = int(scaled_height)
    #                     scaled_width = int(scaled_width)
    #                     estimate_new_resolution = False
                        
    #                 resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
    #                 resized_img.save(os.path.join(rgb_path, file))
    
    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        
        source_image_folder  = os.path.join(self.dataset_source_folder, sequence_name)
        
        timestamps = []
        image_names = []
        for fname in sorted(os.listdir(source_image_folder)):
            if not fname.lower().endswith(".jpg"):
                continue

            ts = self.extract_timestamp_from_filename(fname)
            timestamps.append(ts)
            image_names.append(fname)
        
        with open(rgb_txt, 'w') as f:
            for idx, ts in enumerate(timestamps):
                f.write(f"{ts:.5f} rgb/{image_names[idx]}\n")

    # def create_calibration_yaml(self, sequence_name):
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     rgb_txt = os.path.join(sequence_path, 'rgb.txt')
    #     fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
    #         8154.29, 8154.29, 4076.2906, 2787.575 , -0.0331103, 0.0312571, -0.00133488, 0.00277011, -0.109366)
        
    #     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #     distortion_coeffs = np.array([k1, k2, p1, p2, k3])
        
    #     fx, fy, cx, cy = resize_rgb_images(rgb_txt, sequence_path, 640, 480, camera_matrix)
    #     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #     fx, fy, cx, cy = undistort_rgb_rad_tan(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)

    #     self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)

    # def get_utm_transformer(self, lat, lon):
    #     utm_zone = int((lon + 180) / 6) + 1
    #     hemisphere = "north" if lat >= 0 else "south"
    #     proj_str = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    #     utm_crs = CRS.from_proj4(proj_str)
    #     return Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)

    def extract_timestamp_from_filename(self,filename):
        match = re.search(r'DJI_(\d{14})', filename)
        if match:
            dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
            return dt.timestamp()
        return None

    # def create_groundtruth_txt(self, sequence_name):
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
    #     source_image_folder  = os.path.join(self.dataset_source_folder, sequence_name)

    #     estimate_center = True
    #     with open(groundtruth_txt, 'w') as out_file:
    #         for fname in sorted(os.listdir(source_image_folder)):
    #             if not fname.lower().endswith(".jpg"):
    #                 continue
    #             fpath = os.path.join(source_image_folder, fname)
    #             with open(fpath, 'rb') as f:
    #                 tags = exifread.process_file(f, details=False)

    #             gps_data = self.extract_gps_and_altitude(tags)
    #             ts = self.extract_timestamp_from_filename(fname)

    #             lat, lon, alt = gps_data
    #             print(f"{ts:.3f} {lat:.6f} {lon:.6f} {alt:.3f}")
    #             transformer = self.get_utm_transformer(lat, lon)
    #             x, y = transformer.transform(lon, lat)  # Now in UTM meters
    #             z = alt

    #             if estimate_center:
    #                 center = (x, y)
    #                 estimate_center = False

    #             # x -= center[0]
    #             # y -= center[1]
    #             qx = qy = qz = 0.0
    #             qw = 1.0

    #             out_file.write(f"{ts:.3f} {x:.3f} {y:.3f} {z:.3f} {qx} {qy} {qz} {qw}\n")

    #             #print(x, y, z)
    #             # ts = extract_timestamp_from_filename(fname)
    #             # if gps_data is None or ts is None:
    #             #     print(f"Skipping {fname}")
    #             #     continue

    #     data = np.loadtxt(groundtruth_txt)

    #     # Extract positions
    #     timestamps = data[:, 0]
    #     xs = data[:, 1]
    #     ys = data[:, 2]
    #     zs = data[:, 3]

    #     import matplotlib.pyplot as plt
    #     # Plot 2D trajectory (XY)
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(xs, ys, marker='o', linestyle='-', markersize=2, label='Trajectory')
    #     plt.xlabel("X [m]")
    #     plt.ylabel("Y [m]")
    #     plt.title("2D Trajectory")
    #     plt.axis('equal')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # def remove_unused_files(self, sequence_name):
    #     return
        # sequence_path = os.path.join(self.dataset_path, sequence_name)
        # glomap_results = os.path.join(sequence_path, 'colmap_00000')
        # glomap_keyframe_trajectory = os.path.join(sequence_path, '00000_KeyFrameTrajectory.txt')
        # glomap_calibration_log_file = os.path.join(sequence_path, 'calibration_log_file.txt')
        # glomap_build_log_file = os.path.join(sequence_path, 'glomap_build_log_file.txt')
        #
        # if os.path.exists(glomap_results):
        #     shutil.rmtree(glomap_results)
        #
        # if os.path.exists(glomap_keyframe_trajectory):
        #     os.remove(glomap_keyframe_trajectory)
        # if os.path.exists(glomap_calibration_log_file):
        #     os.remove(glomap_calibration_log_file)
        # if os.path.exists(glomap_build_log_file):
        #     os.remove(glomap_build_log_file)

    # def evaluate_trajectory_accuracy(self, trajectory_txt, groundtruth_txt):
    #     return


    # def dms_to_dd(self, dms, ref):
    #     degrees = dms[0].num / dms[0].den
    #     minutes = dms[1].num / dms[1].den
    #     seconds = dms[2].num / dms[2].den
    #     dd = degrees + minutes / 60.0 + seconds / 3600.0
    #     if ref in ["S", "W"]:
    #         dd = -dd
    #     return dd

    # def extract_gps_and_altitude(self, tags):
    #     try:
    #         lat = self.dms_to_dd(tags["GPS GPSLatitude"].values, tags["GPS GPSLatitudeRef"].printable)
    #         lon = self.dms_to_dd(tags["GPS GPSLongitude"].values, tags["GPS GPSLongitudeRef"].printable)

    #         if tags["GPS GPSLatitudeRef"].printable == "S":
    #             lat = -lat
    #         if tags["GPS GPSLongitudeRef"].printable == "W":
    #             lon = -lon
    #         alt = float(tags.get("GPS GPSAltitude", 0).values[0].num) / tags.get("GPS GPSAltitude", 1).values[0].den
    #     except Exception as e:
    #         print(f"Error extracting GPS: {e}")
    #         return None
    #     return lat, lon, alt