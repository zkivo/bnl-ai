import deeplabcut as dlc
import os
import sys

def check_files(directory, video_files):
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    print(f"Directory '{directory}' exists.")

    # Check if video files exist
    for video_file in video_files:
        if os.path.isfile(video_file):
            print(f"Video file '{video_file}' exists.")
        else:
            print(f"Error: Video file '{video_file}' does not exist.")

if __name__ == "__main__":
    # Ensure there are enough arguments
    if len(sys.argv) < 3:
        print("Usage: python check_files.py <directory_path> <video_file_1> <video_file_2> ...")
        sys.exit(1)

    # Parse arguments
    directory_path = sys.argv[1]
    video_file_paths = sys.argv[2:]

    # Call the check function
    check_files(directory_path, video_file_paths)

    if not directory_path.endswith(os.path.sep):
        directory_path += os.path.sep

    # Combine the directory path and file name
    config_path = directory_path + '\config.yaml'

    dlc.add_new_videos(config_path, video_file_paths, 
                       copy_videos=True,
                       extract_frames=False)


