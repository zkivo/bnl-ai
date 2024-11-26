import sys
import os
import re
import cv2
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.io import imread
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
from collections import defaultdict


def extract_color_histogram(frame):
    """
    This function calculates the color histogram for a given frame.
    The histogram for each color has 8 bins, that means that
    the output will have 8+8+8=24 features.
    This is usuful when the difference between frames is mostly
    seen as color differences.
    """
    frame = cv2.resize(frame, (128, 128))  # Resize for consistency
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extract_hog_features(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (128, 128))
    features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

def group_videos_by_regex(video_paths, group_by):
    # Create a dictionary to store grouped videos
    grouped_videos = defaultdict(list)

    # Compile the regular expression
    regex = re.compile(group_by)

    # Iterate over all video paths and group them
    for path in video_paths:
        match = regex.search(os.path.basename(path))
        if match:
            # Use the matched group as the key for grouping
            group_key = match.group()
            grouped_videos[group_key].append(path)
        else:
            # Videos without a match go into a separate "ungrouped" key
            print(colored(f"Warning: Video '{path}' does not match the group_by pattern.", "yellow"))
            grouped_videos[None].append(path)

    # Convert dictionary values to a list of lists
    grouped_list = [group for group in grouped_videos.values()]
    return grouped_list

def extract_frames(video_list, output_dir, method, n_frames, group_by=None):

    # -----------------------------------------------------------------
    # ------------------- Check input arguments -----------------------
    # -----------------------------------------------------------------
    
    if not isinstance(video_list, list) or not all(isinstance(path, str) for path in video_list):
        print(colored("Error: 'video_list' must be a list of video paths (strings).", "red"))
        sys.exit(1)
    
    # Check all video paths in `video_list` exist
    for path in video_list:
        if not os.path.isfile(path):
            print(colored(f"Error: Video path '{path}' does not exist.", "red"))
            sys.exit(1)

    # Check `output_dir` is a valid directory path
    if not isinstance(output_dir, str) or not os.path.isdir(output_dir):
        print(colored("Error: 'output_dir' must be a valid directory path.", "red"))
        sys.exit(1)
    
    # Check `method` is one of the allowed strings
    if method not in ['linear', 'uniform', 'kmeans']:
        print(colored("Error: 'method' must be one of ['linear', 'uniform', 'kmeans'].", "red"))
        sys.exit(1)
    
    # Check `n_frames` is a positive integer
    if not isinstance(n_frames, int) or n_frames <= 0:
        print(colored("Error: 'n_frames' must be a positive integer.", "red"))
        sys.exit(1)
    
    # Check `group_by` is a valid regular expression or None
    if group_by is not None:
        if not isinstance(group_by, str):
            print(colored("Error: 'group_by' must be a string (regular expression) or None.", "red"))
            sys.exit(1)
        try:
            re.compile(group_by)
        except re.error:
            print(colored("Error: 'group_by' is not a valid regular expression.", "red"))
            sys.exit(1)


    # -----------------------------------------------------------------
    # ------------------------ Group videos  --------------------------
    # -----------------------------------------------------------------=

    
    # Group videos if group_by is provided
    if group_by is not None:
        grouped_videos = group_videos_by_regex(video_list, group_by)
        print(colored(f"Grouped videos based on \"{group_by}\" pattern.", "green"))
        # for i, group in enumerate(grouped_videos):
        #     print(colored(f"Group {i + 1}:", "green"), group)
        # print()
    else:
        # No grouping, treat each video independently
        grouped_videos = [[path] for path in video_list]

    exit()

    for video_file in video_list:
        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Error: Could not open video '{video_file}'.")
            continue

        print("Extracting features with color histogram...")
        frame_count = 0
        frame_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                print(f"frame: {frame_count}", end="\r", flush=True)  # Overwrites the same line and forces flush
            frame_features.append(extract_color_histogram(frame))
            frame_count += 1
        cap.release()

        frame_features = np.array(frame_features)   

        print("Clustering with K-means...")
        # Define the number of clusters (e.g., 25)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(frame_features)

        # Select the frame closest to the centroid of each cluster
        selected_frames = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_features = frame_features[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_frame_index = cluster_indices[np.argmin(distances)]
            selected_frames.append(closest_frame_index)

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Error: Could not open video '{video_file}'.")
            continue

        basename = os.path.basename(video_file)
        root_name, ext = os.path.splitext(basename)

        frames_folder = os.path.join(out_folder, root_name)
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)

        print(f"Writing frames to folder {frames_folder}...")
        frame_number = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % 30 == 0:
                print(f"frame: {frame_number}", end="\r", flush=True)  # Overwrites the same line and forces flush
            if frame_number in selected_frames:
                frame_path = os.path.join(frames_folder, f"{root_name}-{frame_number}.png")
                cv2.imwrite(frame_path, frame)
            frame_number += 1
        cap.release()


def parse_arguments():
    """
    Parse command-line arguments for the extract_frames.py script, which extracts 
    frames from videos based on the specified method and number of frames.

    Arguments:
        - `input`: Either a list of one or more video file paths or a directory containing videos.
        - `output-dir`: Path to the directory where extracted frames will be saved.
        - `method`: The method to use for extracting frames, with options:
            - `linear`: Extract frames evenly spaced across the video duration.
            - `uniform`: Extract frames such that the time interval between frames is constant.
            - `kmeans`: Use clustering to select frames that represent key moments in the video.
        - `n-frames`: Total number of frames to extract from each video.
        - `group-by`: A regex pattern specifying how to group videos recorded simultaneously.
        - `recursive`: Whether to search directories recursively for video files.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = \
        argparse.ArgumentParser(description="Extract frames from video files and save them to a specified directory. "
                                            "Supports grouping of videos recorded simultaneously using a regex pattern.")
    
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        required=True,
        help=(
            "Either a list of video file paths or a directory containing video files. "
            "If a directory is provided, videos in the directory will be processed. "
            "To specify multiple video files, provide a space-separated list."
        )    
    )
    
    # Recursive flag for directory input
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search directories recursively for video files if input is a directory.",
    )

    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Path to the directory where the extracted frames will be saved. "
             "The directory should exist or will be created if it does not."
    )
    
    parser.add_argument(
        "-m", "--method",
        choices=["linear", "uniform", "kmeans"],
        default="linear",
        help="Method for extracting frames:\n"
            "  - `linear`: Extract frames evenly spaced across the video.\n"
            "  - `uniform`: Extract frames randomly with uniform distribution.\n"
            "  - `kmeans`: Use clustering to select representative frames.\n"
            "Default is 'linear'."
    )
    
    def positive_integer(value):
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")

    parser.add_argument(
        "-n", "--n-frames",
        type=positive_integer,
        required=True,
        help="Total number of frames to extract. This number "
             "must be a positive integer."
    )

    def valid_regex(value):
        try:
            re.compile(value)
            return value
        except re.error:
            raise argparse.ArgumentTypeError(f"Invalid regex: {value}. Please provide a valid regular expression.")


    # Group-by: Regex for grouping videos
    parser.add_argument(
        "-g", "--group-by",
        type=valid_regex,
        required=False,
        help=(
            "A regex pattern to specify how to group videos recorded simultaneously (e.g., different camera angles). "
            "This is helpful when using 'kmeans' or 'uniform' methods, where grouped videos are treated as a single entity "
            "for frame extraction. If not provided, each video is treated individually."
        ),
    )
    
    return parser.parse_args()

def get_video_files(input_path, recursive):
    """
    Retrieve video files from the input path.

    Args:
        input_path (str): Either a directory or a list of video files.
        recursive (bool): Whether to search recursively in directories.

    Returns:
        list: A list of video file paths.

    Raises:
        argparse.ArgumentTypeError: If the input path is invalid or no video files are found.
    """
    if os.path.isdir(input_path):
        pattern = "**/*" if recursive else "*"
        video_files = [str(p) for p in Path(input_path).glob(pattern) if p.is_file()]
        if not video_files:
            raise argparse.ArgumentTypeError(
                f"No video files found in directory: {input_path} (recursive={recursive})"
            )
        return video_files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid input: {input_path}. Must be a valid file or directory."
        )

if __name__ == "__main__":

    args = parse_arguments()
    print(args.input, args.recursive, args.output_dir, args.method, args.n_frames, args.group_by)

    # Handle input videos
    input_videos = []
    for item in args.input:
        input_videos.extend(get_video_files(item, args.recursive))

    if args.group_by is None:
        print(colored("Warning: 'group_by' is not specified. Processing videos individually.", "yellow"))

    extract_frames(input_videos, args.output_dir, args.method, args.n_frames, args.group_by)
