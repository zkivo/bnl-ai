import cv2
import random
import os

def extract_random_frames(video_path, num_frames, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # files a corrupted.
    # to repair it can be used ffmpeg with linux
    print(video_path, total_frames)

    # Randomly select frame indices
    frame_indices = random.sample(list(range(total_frames)), num_frames)

    # Sort the indices to read frames sequentially for efficiency
    frame_indices.sort()

    extracted_frames = 0
    current_frame = 0

    while cap.isOpened() and extracted_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is one of the randomly selected ones
        if current_frame in frame_indices:
            frame_filename = os.path.join(output_dir, f"frame_{current_frame}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_frames += 1
            print(f"Extracted frame {current_frame} to {frame_filename}")

        current_frame += 1

    # Release the video capture object
    cap.release()

# Usage example
video_path = 'data/video-18-10-47_2.mkv'
num_frames = 5  # Number of random frames you want to extract
output_dir = 'output'  # Directory to save the frames

extract_random_frames(video_path, num_frames, output_dir)
