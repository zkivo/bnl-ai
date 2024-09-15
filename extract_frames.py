import tempfile
import sys
import os
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():
    # Check if two arguments were provided
    if len(sys.argv) < 3:
        print(bcolors.FAIL + "Error: Two arguments are required.\n" + bcolors.ENDC + \
              "1) dir path which contains the videos where to extract the frames.\n" \
              "This script will recursively search into the dirs.\n" \
              "2) Regex patter for matching the video names.")
        sys.exit(1)

    # Get the arguments
    video_dir  = sys.argv[1]
    regex = sys.argv[2]

    if not os.path.isdir(video_dir):
        print(f"Error: The path '{video_dir}' is not a dir.")
        sys.exit(1)

    try:
        re.compile(regex)
        print(f"Regex pattern '{regex}' is valid.")
    except re.error:
        print(f"Error: The regex pattern '{regex}' is not valid.")
        sys.exit(1)

    print(bcolors.OKGREEN, "---- Part 1 ----", bcolors.ENDC)

    temp_dir = tempfile.gettempdir()

    print("temp_dir: ", temp_dir)

    videos = []

    # Walk through the directory recursively
    for dirpath, dirnames, files in os.walk(video_dir):
        for file in files:
            if re.search(regex, file):
                videos.append(os.path.join(dirpath, file))

    print("videos: ", videos)

    if len(videos) == 0:
        print(bcolors.FAIL + f"No videos found in '{video_dir}' with the regex pattern '{regex}'." + bcolors.ENDC)
        sys.exit(1)

    print(bcolors.OKGREEN, "---- Part 2 ----", bcolors.ENDC)
    import deeplabcut as dlc

    print("creating new project...")
    config_file = dlc.create_new_project('-', '-', videos,
                                         working_directory=temp_dir,
                                         copy_videos=False, multianimal=False)

    print("path of config file:", os.path.dirname(config_file))
    print("extracting frames...")

    try:
        dlc.extract_frames(config_file, 'automatic', 'kmeans', crop=False, userfeedback=False)
    except ValueError as e:
        print(bcolors.FAIL + "Error: ", e, bcolors.ENDC)
        if str(e) == "__len__() should return >= 0":
            print("fixing corrupted videos with ffmpeg...")
            # fix corrupted video with ffmpeg

        sys.exit(1)
    except Exception as e:
        print(bcolors.FAIL + "Error: ", e, bcolors.ENDC)
        sys.exit(1)

if __name__ == "__main__":
    main()
