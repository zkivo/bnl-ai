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

    temp_dir = tempfile.gettempdir()

    print("temp_dir: ", temp_dir)

    videos = []

    # Walk through the directory recursively
    for dirpath, dirnames, files in os.walk(video_dir):
        for file in files:
            if re.search(regex, file):
                videos.append(os.path.join(dirpath, file))

    print("videos: ", videos)

if __name__ == "__main__":
    main()
