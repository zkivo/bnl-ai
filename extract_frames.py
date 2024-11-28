import sys
import os
import re
import argparse
import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from termcolor import colored
from pathlib import Path
from collections import defaultdict

def contours_extraction(video_paths, 
                        output_folder,
                        n_clusters=50, 
                        warm_up_frames=100, 
                        show_visuals=True):
    """
    Extracts contours from multiple videos and applies 
    K-Means clustering to select representative frames.
    
    Parameters
    ----------
    video_paths : list of str
        List of paths to video files from different cameras
    output_folder : str
        Folder to save the selected frames
    n_clusters : int, optional
        Number of clusters for KMeans clustering, by default 50
    warm_up_frames : int, optional
        Number of frames to train the background model, by default 100
    show_visuals : bool, optional
        Whether to show visualizations, by default True
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_paths.sort()  # Sort video paths for consistency

    # Initialize background subtractors for each camera
    background_subtractors = [cv2.createBackgroundSubtractorMOG2() for _ in video_paths]

    # Open video captures for all cameras
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    # Train background model with initial frames (warm-up period)
    print(f"Training background models with first {warm_up_frames} frames ...")
    for frame_idx in range(warm_up_frames):
        for cap, back_sub in zip(caps, background_subtractors):
            ret, frame = cap.read()
            if not ret:
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            back_sub.apply(gray_frame)  # Update the background model

    print("Background models trained. Starting feature extraction...")

    # Extract features for all remaining frames
    frame_features = []
    end_video = False
    while not end_video:
        if frame_idx > 500:
            break
        combined_features = []
        if frame_idx % 30 == 0:
            print(f"frame: {frame_idx}", end="\r", flush=True)
        for cam_idx, (cap, back_sub) in enumerate(zip(caps, background_subtractors)):
            ret, frame = cap.read()
            if not ret:
                end_video = True
                continue
            # Apply background subtraction for this camera
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = back_sub.apply(gray_frame)

            # Apply dilation to make the foreground mask larger
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(fg_mask, kernel, iterations=3)

            # breakpoint()
            # Visualize background subtraction result
            if show_visuals and frame_idx % 100 == 0:  # Show visuals every 100 frames
                cv2.imshow(f"Bsckground Mask - Camera {cam_idx}", back_sub.getBackgroundImage())
                cv2.imshow(f"Dilatated Mask - Camera {cam_idx}", dilated_mask)
                # cv2.imshow(f"Foreground Mask - Camera {cam_idx}", fg_mask)
                # cv2.imshow(f"Original Frame - Camera {cam_idx}", gray_frame)
                cv2.waitKey(1)  # Wait for 100ms

            # Find contours of the foreground (mouse)
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use the largest contour (assume it's the mouse)
                largest_contour = max(contours, key=cv2.contourArea)
                # Apply convex hull for a more inclusive contour
                hull = cv2.convexHull(largest_contour)
                # Extract features: area, perimeter, centroid, bounding box
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(largest_contour, True)
                moments = cv2.moments(largest_contour)
                centroid_x = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
                centroid_y = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
                bounding_box = cv2.boundingRect(largest_contour)

                # Draw bounding box and centroid on the frame for visualization
                if show_visuals and frame_idx % 100 == 0:
                    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (255, 0, 0), -1)
                    cv2.imshow(f"Processed Frame - Camera {cam_idx}", frame)
                    cv2.waitKey(1)

                # Create a feature vector
                feature_vector = [area, perimeter, centroid_x, centroid_y, *bounding_box]
                combined_features.append(feature_vector)

        # Aggregate features from all cameras
        if len(combined_features) == len(caps):  # Ensure all cameras have data
            aggregated_features = np.mean(combined_features, axis=0)
            frame_features.append((frame_idx, aggregated_features))
        frame_idx += 1

    # Prepare features for clustering
    frame_indices, feature_vectors = zip(*frame_features)
    feature_matrix = np.array(feature_vectors)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_matrix)

    # Select the frame closest to the centroid of each cluster
    selected_frames = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_features = feature_matrix[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_frame_index = cluster_indices[np.argmin(distances)]
        selected_frames.append(closest_frame_index)

    # Release all captures
    for cap in caps:
        cap.release()

    # Re-open video captures for all cameras
    # Save the selected frames into the output folder
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    j = 0
    for cap in caps:
        i = 0
        basename = os.path.basename(video_paths[j])
        root_name, ext = os.path.splitext(basename)
        frames_folder = os.path.join(output_folder, root_name)
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in selected_frames:
                output_path = os.path.join(frames_folder, f"{root_name}-{i}.png")
                cv2.imwrite(output_path, frame)
            i += 1
        j += 1

    print(f"Extracted frames saved in {output_folder}")
    cv2.destroyAllWindows()


def uniform_extraction(video_paths, output_dir, n_frames, random = True):
    """
    Extracts n_frames from each video in video_paths with a uniform 
    random distribution or linearly spaced (if random = false).

    Args:
        video_paths (list): List of paths to video files.
        output_dir (str): Directory where extracted frames will be saved.
        n_frames (int): Number of frames to extract from each video.
        random (bool): Whether to sample frames randomly or linearly.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Counting total frames...")
    cap = cv2.VideoCapture(video_paths[0])
    total_frames = 0
    while True:
        if total_frames % 30 == 0:
            print(f"total_frames: {total_frames}", end="\r", flush=True)
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
    cap.release()

    # Uniformly sample `n_frames` frame indices
    if random:
        frame_indices = sorted(random.sample(range(total_frames), n_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    print("Frame indices to extract:", frame_indices)

    # Process each video
    j = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        i = 0
        basename = os.path.basename(video_paths[j])
        root_name, ext = os.path.splitext(basename)
        frames_folder = os.path.join(output_dir, root_name)
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)
        while True:
            if i % 30 == 0:
                print(f"frames: {i}", end="\r", flush=True)
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                output_path = os.path.join(frames_folder, f"{root_name}-{i}.png")
                cv2.imwrite(output_path, frame)
            i += 1
        cap.release()
        j += 1

def linear_extraction(video_path, output_folder, interval : float = 1000):
    """
    Extract frames linearly from a video file taken the distance in milliseconds.
    Then save the frames in an output folder.

    Arguments
    -------

    video_path: str
        Path to the video file.
    output_folder: str
        Path to the folder where the frames will be saved.
    interval: int
        Interval (ms) between frames to be saved. 
        For example, 1000 means 1 frame per second.

    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Get the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Could not determine the FPS of the video {video_path}.")
        return

    # milliseconds to seconds
    interval = float(interval) / float(1000)
    # Calculate the frame interval
    frame_interval = int(fps * interval)
    frame_count = 0
    basename = os.path.basename(video_path)
    root_name, ext = os.path.splitext(basename)
    frames_folder = os.path.join(output_folder, root_name)
    os.makedirs(frames_folder, exist_ok=True)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Save the frame if it matches the interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(frames_folder, f"{root_name}-{frame_count}.png")
            cv2.imwrite(output_path, frame)
        frame_count += 1
    video.release()
    print(f"Frames saved in folder {frames_folder}")

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

def extract_frames(video_list, output_dir, method, n_frames, time_interval, group_by=None):

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
    if method not in ['linear', 'uniform', 'contours']:
        print(colored("Error: 'method' must be one of ['linear', 'uniform', 'contours'].", "red"))
        sys.exit(1)
    
    # Check `n_frames` is a positive integer
    if n_frames is not None:
        if not isinstance(n_frames, int) or n_frames <= 0:
            print(colored("Error: 'n_frames' must be a positive integer.", "red"))
            sys.exit(1)

    # Check `n_frames` is a positive integer
    if time_interval is not None:
        if not isinstance(time_interval, int) or time_interval <= 0:
            print(colored("Error: 'time_interval' must be a positive integer.", "red"))
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
    # -----------------------------------------------------------------

    
    # Group videos if group_by is provided
    if group_by is not None:
        grouped_videos = group_videos_by_regex(video_list, group_by)
        print(colored(f"Grouped videos based on \"{group_by}\" pattern.", "green"))
    else:
        # No grouping, treat each video independently
        grouped_videos = [[path] for path in video_list]

    for i, group in enumerate(grouped_videos):
        print(colored(f"Group {i + 1}:", "green"), group)

        if method == "contours":
            contours_extraction(group, output_dir, n_frames)
        elif method == "uniform":
            uniform_extraction(group, output_dir, n_frames)
        elif method == "linear":
            for video_path in group:
                linear_extraction(video_path, output_dir)


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
            - `contours`: Use clustering with countours extraction technique to select frames that represent key moments in the video.
        - `n-frames`: Total number of frames to extract from each video.
        - `group-by`: A regex pattern specifying how to group videos recorded simultaneously.
        - `recursive`: Whether to search directories recursively for video files.
        - `time-interval`: Time in milliseconds between frames for `linear` method.

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
        choices=["linear", "uniform", "contours"],
        default="linear",
        help="Method for extracting frames:\n"
            "  - `linear`: Extract frames evenly spaced across the video.\n"
            "  - `uniform`: Extract frames randomly with uniform distribution.\n"
            "  - `contours`: Use clustering with contours extraction to select representative frames.\n"
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
        required=False,
        help="Total number of frames to extract. This number "
             "must be a positive integer."
    )

    # Time interval: Optional for linear method
    parser.add_argument(
        "-t", "--time-interval",
        type=positive_integer,
        required=False,
        help=(
            "Time interval in milliseconds between frames. Only applicable when using the 'linear' method. "
            "Ignored for other methods."
        ),
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
            "This is helpful when using 'contours' or 'uniform' methods, where grouped videos are treated as a single entity "
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
    print(args.input, args.recursive, args.output_dir, args.method, args.n_frames, args.time_interval, args.group_by)

    if args.method != "linear" and args.time_interval is not None:
        print(colored("--time-interval is only valid when the method is 'linear'.", "red"))
        exit(1)

    if args.method != "linear" and args.n_frames is None:
        print(colored("--n-frames is required when method is not 'linear'.", "red"))
        exit(1)
    
    if args.time_interval is not None and args.n_frames is not None:
        print(colored("--n-frames and --time-interval cannot be used together. Choose one.", "red"))
        exit(1)
    
    if args.method == "linear" and args.n_frames is None and args.time_interval is None:
        print(colored("Please insert either --n-frames or --time-interval", "red"))
        exit(1)

    # Handle input videos
    input_videos = []
    for item in args.input:
        input_videos.extend(get_video_files(item, args.recursive))

    if args.group_by is None:
        print(colored("Warning: 'group_by' is not specified. Processing videos individually.", "yellow"))

    extract_frames(args.input, args.output_dir, args.method, args.n_frames, args.time_interval, args.group_by)
