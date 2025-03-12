from ultralytics import YOLO
import argparse
import torch
import yaml
import hrnet
import os
import glob

def get_image_files(input_path):
    filenames = []
    
    if os.path.isdir(input_path):
        # Get all image files in the directory (common image formats)
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
        for ext in image_extensions:
            filenames.extend(glob.glob(os.path.join(input_path, ext)))
    elif os.path.isfile(input_path):
        # If it's a single file, check if it's an image
        if input_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
            filenames.append(input_path)
    else:
        # If it's a list of files, check and add valid image files
        potential_files = input_path.split(',')
        for file in potential_files:
            file = file.strip()
            if os.path.isfile(file) and file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
                filenames.append(file)
    
    return filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image files from a folder, list, or single file.")
    parser.add_argument("-i", "--input", required=True, help="Input folder, list of image files (comma-separated), or single image file")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    filenames = get_image_files(args.input)
    print("Collected image files:", filenames)

    model_detection = YOLO(r"runs\detect\train15\weights\best.pt")  # pretrained YOLO11n model

    with open(r'config\hrnet_w48_384_288.yaml', 'r') as f:
        cfg_w48_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 14
        model_pose = hrnet.get_pose_net(cfg_w48_384_288)
        model_pose.load_state_dict(torch.load(r'out\train-250307_154813\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))

    results = model_detection(filenames)  # return a list of Results objects


    # Process results list
    for result in results:
        bbox = result.boxes.xyxy[0]