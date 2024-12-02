import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
import os
from tqdm import tqdm

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

	# File paths
	sequence_path = '/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0'
	rgb_txt = '/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0/rgbd_depth_anything_v2.txt'
	#rgb_txt = '/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0/rgb.txt'

	rgb_paths = []
	with open(rgb_txt, 'r') as file:
		lines = file.readlines()
		for line in lines:
			parts = line.strip().split()
			rgb_file = os.path.join(sequence_path, parts[1])
			rgb_path = os.path.splitext(os.path.basename(rgb_file))[0]
			rgb_paths.append(rgb_path)

	scales = []
	intercepts = []
	for filename in tqdm(rgb_paths):
		image_path = f'/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0/depth/{filename}.png'
		image_path2 = f'/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0/depth_anything_v2/{filename}.npy'
		#image_path2 = f'/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office0/depth_anything_v2/{filename}.png'

		# Load the depth image from PNG
		depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 6553.5
		if depth_image is None:
			raise FileNotFoundError(f"Depth image file not found: {image_path}")

		#non_zero_values = depth_image[depth_image > 0]
		#max_value = np.max(non_zero_values)
		#print(f"Max value of depth_image (ignoring zeros): {max_value}")

		# Load the depth image from NumPy file
		#depth_image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED) / 6553.5
		depth_image2 = np.load(image_path2)

		# Invert depth_image2
		#depth_image2 = 1.0 / depth_image2

		# Flatten both images to apply regression on non-zero pixel values
		valid_mask = (depth_image > 0) & (depth_image2 > 0) # Exclude zero values in depth_image
		depth_image_flat = depth_image[valid_mask].reshape(-1, 1)
		depth_image2_flat = depth_image2[valid_mask].reshape(-1, 1)

		# Subsample data for memory efficiency (if dataset is large)
		subsample_size = 1000  # Adjust as needed for memory constraints
		if len(depth_image_flat) > subsample_size:
			indices = np.random.choice(len(depth_image_flat), subsample_size, replace=False)
			depth_image_flat = depth_image_flat[indices]
			depth_image2_flat = depth_image2_flat[indices]

		# Apply robust linear regression (Theil-Sen Estimator)
		#regressor = TheilSenRegressor(random_state=42)
		#regressor.fit(depth_image2_flat, depth_image_flat.ravel())  # Flatten y using ravel()
		#scaled_depth_image2 = regressor.predict(depth_image2.flatten().reshape(-1, 1)).reshape(depth_image2.shape)

		inv_depth_image_flat = 1.0 / depth_image_flat
		inv_depth_image2_flat = 1.0 / depth_image2_flat
		depth_image2[depth_image2 < 0.0000001] = -1.0
		inv_depth_image2 = 1.0 / depth_image2
		inv_depth_image2[inv_depth_image2 < 0.0] = 0.0

		regressor = TheilSenRegressor(random_state=42)
		print("inv_depth_image2_flat")
		analyze_image(inv_depth_image2_flat)
		print("inv_depth_image_flat")
		analyze_image(inv_depth_image_flat)
		regressor.fit(inv_depth_image2_flat, inv_depth_image_flat.ravel())  # Flatten y using ravel()
		scaled_inv_depth_image2 = regressor.predict(inv_depth_image2.flatten().reshape(-1, 1)).reshape(
			inv_depth_image2.shape)
		scaled_depth_image2 = 1.0 / scaled_inv_depth_image2

		scales.append(regressor.coef_[0])
		intercepts.append(regressor.intercept_)

		# Print regression parameters
		print(f"Regression Coefficients:")
		print(f"  Slope: {regressor.coef_[0]:.4f}")
		print(f"  Intercept: {regressor.intercept_:.4f}")

		# Check shapes
		if depth_image.shape != scaled_depth_image2.shape:
		    print("The dimensions of the images are different:")
		    print(f"PNG Image shape: {depth_image.shape}")
		    print(f"NumPy Image shape: {scaled_depth_image2.shape}")
		else:
		    print(f"Both images have the same shape: {depth_image.shape}")

		# Compare images
		difference = np.abs(depth_image.astype(np.float32) - scaled_depth_image2.astype(np.float32))
		#difference = 1.0/np.abs((1.0/depth_image.astype(np.float32)) - (1.0/scaled_depth_image2.astype(np.float32)))
		max_difference = np.max(difference)
		mean_difference = np.mean(difference)

		print(f"Maximum pixel difference: {max_difference}")
		print(f"Mean pixel difference: {mean_difference}")

		# Determine shared color scale limits
		shared_vmin = min(depth_image.min(), depth_image2.min())
		shared_vmax = max(depth_image.max(), depth_image2.max())

		scaled_depth_image2 = scaled_depth_image2 * 6553.5
		scaled_depth_image2 = scaled_depth_image2.astype(np.uint16)
		image_path2 = image_path2.replace('npy', 'png')
		cv2.imwrite(image_path2, scaled_depth_image2)

		# Visualize the results
		# plt.figure(figsize=(15, 5))
		#
		# # Original PNG image
		# plt.subplot(1, 3, 1)
		# plt.title("Depth Image (PNG)")
		# plt.imshow(depth_image, cmap='viridis', vmin=shared_vmin, vmax=shared_vmax)
		# plt.colorbar()
		#
		# # Scaled NumPy depth image
		# plt.subplot(1, 3, 2)
		# plt.title("Scaled Depth Image (NumPy, Robust Regression)")
		# plt.imshow(scaled_depth_image2, cmap='viridis', vmin=shared_vmin, vmax=shared_vmax)
		# plt.colorbar()
		#
		# # Difference
		# plt.subplot(1, 3, 3)
		# plt.title("Difference")
		# plt.imshow(difference, cmap='hot')  #, vmin=0, vmax=1)
		# plt.colorbar()
		#
		# plt.tight_layout()
		# plt.show()

# Scales histogram
plt.subplot(1, 2, 1)
plt.hist(scales, bins=10, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Scales (Slopes)')
plt.xlabel('Scale (Slope)')
plt.ylabel('Frequency')

# Intercepts histogram
plt.subplot(1, 2, 2)
plt.hist(intercepts, bins=10, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogram of Intercepts')
plt.xlabel('Intercept')
plt.ylabel('Frequency')

# Show plots
plt.tight_layout()
plt.show()