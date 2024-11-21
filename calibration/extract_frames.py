import cv2
import os

def extract_frames_from_video(video_path, output_folder, interval=2):
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

    # Calculate the frame interval
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame if it matches the interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    video.release()
    print(f"Frames saved for video {video_path} in folder {output_folder}")

def process_videos_in_folder(input_folder, interval=2):
    # List all files in the input folder
    for filename in os.listdir(input_folder):
        video_path = os.path.join(input_folder, filename)
        # Check if the file is a video file (you can add more extensions if needed)
        if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Create output folder named after the video (without extension)
            video_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(input_folder, video_name)
            extract_frames_from_video(video_path, output_folder, interval)

# Example usage
if __name__ == "__main__":
    input_folder = "data"  # Replace with the path to your videos folder
    process_videos_in_folder(input_folder)
