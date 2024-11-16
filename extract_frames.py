import deeplabcut as dlc
import time
import threading
import os
import sys

# TODO: Add skip/skip all functionality

if len(sys.argv) != 2:
    print("Usage: python extract_frames.py <directory_path>")
    sys.exit(1)

# Parse arguments
directory_path = sys.argv[1]

if not directory_path.endswith(os.path.sep):
    directory_path += os.path.sep

if not os.path.isdir(directory_path):
    print(f"Error: Directory '{directory_path}' does not exist.")

print(f"Directory '{directory_path}' exists.")

config_path = directory_path + 'config.yaml'
videos_path = directory_path + 'videos'
labeled_data_path = directory_path + 'labeled-data'
video_filenames = [f for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f))]
skip_all = False

user_choice = None

for filename in video_filenames:
    dir_path = os.path.join(labeled_data_path, os.path.basename(filename))
    if os.path.isdir(dir_path):
        if skip_all:
            print(f"Skipping {filename} (skip_all is active)")
            continue

        def user_input():
            nonlocal user_choice
            user_choice = input(f"Directory '{dir_path}' exists. Skip this? (Y/n/(a)ll): ").strip().lower()

        thread = threading.Thread(target=user_input)
        thread.start()

        thread.join(timeout=10)  # Wait for user input or timeout

        if thread.is_alive():
            print("\nNo response. Skipping automatically.")
            thread.join()  # Ensure the thread exits cleanly
            continue

        if user_choice == "y" or user_choice == "yes":
            print(f"Skipping {filename}.")
            continue
        elif user_choice == "a" or user_choice == "all":
            print("Skipping all future directories.")
            skip_all = True
            continue
        elif user_choice == "n" or user_choice == "no":
            print(f"Processing {filename}.")
            pass  # Do your operation here
        else:
            print("Invalid response. Skipping this directory.")
            continue
    else:
        print(f"Directory '{dir_path}' does not exist. Performing operation.")
        pass  # Do your operation here


dlc.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False)
