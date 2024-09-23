import argparse
import os

# Label for script-specific outputs
SCRIPT_LABEL = f"[{os.path.basename(__file__)}] "


def downsample_rgb(timestamps, rgb_paths, step, max_count):
    selected_rgb_paths = []
    selected_timestamps = []

    index = 0
    while index < len(rgb_paths):
        selected_rgb_paths.append(rgb_paths[int(index)])
        selected_timestamps.append(timestamps[int(index)])
        index += step

    # Ensure the number of selected images does not exceed the maximum allowed
    if len(selected_rgb_paths) > max_count:
        selected_rgb_paths = selected_rgb_paths[:int(max_count)]
        selected_timestamps = selected_timestamps[:int(max_count)]

    return selected_rgb_paths, selected_timestamps


def downsample_rgb_frames(rgb_txt, max_rgb_count, min_fps, verbose=False):
    # Read timestamps and paths from rgb.txt
    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path = line.strip().split(' ')
            rgb_paths.append(path)
            rgb_timestamps.append(float(timestamp))

    # Determine downsampling parameters
    if verbose:
        print(f"\n{SCRIPT_LABEL} Processing file: {rgb_txt}")
        print("\nDownsampling settings:")
        print(f"  Maximum number of RGB images: {max_rgb_count}")
        print(f"  Minimum FPS: {min_fps:.1f} Hz")

    sequence_duration = rgb_timestamps[-1] - rgb_timestamps[0]
    actual_fps = len(rgb_paths) / sequence_duration
    max_interval = 1.0 / min_fps
    min_interval = sequence_duration / max_rgb_count

    if min_interval < max_interval:
        max_interval = min_interval
        if verbose:
            print(f"  Adjusted FPS to: {1.0 / max_interval:.2f} Hz")

    step_size = max_interval * actual_fps
    if step_size < 1:
        step_size = 1

    if verbose:
        print(f"  Step size: {step_size:.2f}")

    # Downsample RGB images
    downsampled_paths, downsampled_timestamps = downsample_rgb(rgb_timestamps, rgb_paths, step_size, max_rgb_count)
    downsampled_duration = downsampled_timestamps[-1] - downsampled_timestamps[0]
    downsampled_fps = len(downsampled_paths) / downsampled_duration

    if verbose:
        print("\nDownsampling RGB images:")
        print(f"  Sequence duration: {downsampled_duration:.2f} / {sequence_duration:.2f} seconds")
        print(f"  Number of RGB images: {len(downsampled_paths)} / {len(rgb_paths)}")
        print(f"  RGB frequency: {downsampled_fps:.2f} / {actual_fps:.2f} Hz")

    return downsampled_paths, downsampled_timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument('sequence_path', type=str, help="Path to the sequence directory.")
    parser.add_argument('--rgb_txt', type=str, default="rgb.txt", help="Filename of the input RGB list.")
    parser.add_argument('--rgb_ds_txt', type=str, default="rgb_ds.txt", help="Filename for the downsampled RGB list.")
    parser.add_argument('--max_rgb', type=int, default=10, help="Maximum number of RGB images.")
    parser.add_argument('--min_fps', type=float, default=10.0, help="Minimum downsampled frames per second.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")

    args = parser.parse_args()

    rgb_txt = os.path.join(args.sequence_path, args.rgb_txt)
    rgb_ds_txt = os.path.join(args.sequence_path, args.rgb_ds_txt)
    max_rgb = args.max_rgb
    min_fps = args.min_fps

    downsampled_paths, downsampled_timestamps = downsample_rgb_frames(rgb_txt, max_rgb, min_fps, args.verbose)

    # Write downsampled RGB data to file
    print(f"\nWriting downsampled RGB list to: {rgb_ds_txt}")
    with open(rgb_ds_txt, 'w') as file:
        for timestamp, path in zip(downsampled_timestamps, downsampled_paths):
            file.write(f"{timestamp} {path}\n")
