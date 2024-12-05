import deeplabcut as dlc
import os
import argparse
from termcolor import colored

def validate_inputs(root_path, videos):
    """
    Validate that the root_path is a directory and the videos are .mp4 files.
    """
    if not os.path.isdir(root_path):
        raise ValueError(f"The root project path '{root_path}' is not a valid directory.")
    for video in videos:
        if not video.endswith('.mp4'):
            raise ValueError(f"The file '{video}' is not a valid .mp4 file.")
        if not os.path.isfile(video):
            raise ValueError(f"The file '{video}' does not exist.")

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process videos with root project path.")
    parser.add_argument('root_path', type=str, help="Root project path")
    parser.add_argument('videos', type=str, nargs='+', help="List of video file paths")
    parser.add_argument('-s', '--skip', action='store_true', help="Skip processing flag")
    
    args = parser.parse_args()
    
    try:
        validate_inputs(args.root_path, args.videos)
    except ValueError as e:
        print(f"Input validation error: {e}")
        return

    print(f"Root project path: {args.root_path}")
    print(f"Videos: {args.videos}")
    print(f"Skip flag: {args.skip}")

    config_path =  os.path.join(args.root_path, 'config.yaml')
    labeled_data_path = os.path.join(args.root_path, 'labeled-data')
    extracted_videos = [os.path.basename(dir) for dir in os.listdir(labeled_data_path)]

    for video_input in args.videos:
        root = os.path.splitext(os.path.basename(video_input))[0]
        # if not args.skip:
        #     if root in extracted_videos:
        #         print(colored(f"Warning: {root} alreay extracted. Please use the --skip option to skip this check.", "yellow"))
        #         continue
        print(f"Extracting frames from {video_input}...")
        dlc.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False, videos_list=[video_input])

if __name__ == '__main__':
    main()
