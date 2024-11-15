import deeplabcut as dlc
import os
import sys

# TODO: Add skip/skip all functionality

if len(sys.argv) != 2:
    print("Usage: python extract_frames.py <directory_path>")
    sys.exit(1)

# Parse arguments
directory_path = sys.argv[1]

if not os.path.isdir(directory_path):
    print(f"Error: Directory '{directory_path}' does not exist.")

print(f"Directory '{directory_path}' exists.")

if not directory_path.endswith(os.path.sep):
    directory_path += os.path.sep

# Combine the directory path and file name
config_path = directory_path + '\config.yaml'

dlc.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False)
