import json
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from torchvision.transforms.functional import resize
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import torch
import zipfile
import numpy as np
import argparse
import torch
import glob
import yaml
import SHG
import hrnet
import cv2
import os
import glob

def extract_keypoints_with_confidence(heatmaps):
    """
    Extracts keypoints and their confidence from heatmaps.
    Args:
        heatmaps: Tensor of shape (num_keypoints, h, w) containing heatmaps for each keypoint.
    Returns:
        keypoints_with_confidence: List of ((x, y), confidence) for each keypoint.
    """
    keypoints_with_confidence = []
    for heatmap in heatmaps:
        heatmap = heatmap.squeeze(0)
        # Get the maximum value and its index
        max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
        y, x = divmod(max_idx.item(), heatmap.size(1))  # Convert linear index to x, y
        
        # Confidence as the maximum value
        confidence = max_val.item()
        keypoints_with_confidence.append(((x, y), confidence))
    return keypoints_with_confidence

def create_zip(zip_file_path, folder_path, file_paths):
    """
    Creates a ZIP file containing the given folder and additional files.

    :param zip_file_path: Path for the output ZIP file.
    :param folder_path: Path to the folder to be added to the ZIP.
    :param file_paths: List of file paths to be included in the ZIP.
    """
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add individual files
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"Warning: {file} not found.")

        # Add folder contents if folder exists
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(folder_path))  # Preserve folder structure
                    zipf.write(file_path, arcname)
        else:
            print(f"Warning: Folder '{folder_path}' not found.")

    print(f"ZIP file created successfully: {zip_file_path}")

def calculate_padding(original_width, original_height, target_width, target_height):
    original_width, original_height = target_height, target_width
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        # Pad height
        new_height = original_width / target_aspect_ratio
        padding_height = (new_height - original_height) / 2
        padding_width = 0
    else:
        # Pad width
        new_width = original_height * target_aspect_ratio
        padding_width = (new_width - original_width) / 2
        padding_height = 0

    return int(padding_width), int(padding_height)


def get_image_files(input_path, recursive=False):
    filenames = []
    
    if os.path.isdir(input_path):
        # Get all image files in the directory (and subdirectories if recursive=True)
        image_extensions = ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')
        pattern = "**/*" if recursive else "*"
        for ext in image_extensions:
            filenames.extend(glob.glob(os.path.join(input_path, pattern + "." + ext), recursive=recursive))
    
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
    

def add_annotation_CVAT(index, filename, keypoints, bbox, output_folder, data_folder):
    basename = os.path.splitext(os.path.basename(filename))[0]
    annotation_file = os.path.join(output_folder, 'annotations.json')
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            annotation_base = json.load(f)
    else:
        annotation_base = [{"version": 0, "tags": [], "shapes": [], "tracks": []}]
    skeleton_base = {
        "type": "skeleton",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [],
        "frame": index,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "elements": [],
        "label": "TopSkeleton"
    }
    element_base = {
        "type": "points",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": [],
        "frame": index,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "label": ""
    }
    rectangle_base = {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": bbox.tolist(),
        "frame": index,
        "group": 0,
        "source": "manual",
        "attributes": [],
        "elements": [],
        "label": "Mouse"
    }
    annotation_base[0]["shapes"].append(rectangle_base)
    labels = {'back_croup': 0, 'back_midpoint': 1, 'back_withers': 2, 'ears_midpoint': 3, 'left_ear_base': 4, 'left_ear_tip': 5, 'nose': 6, 'right_ear_base': 7, 'right_ear_tip': 8, 'tail_base': 9, 'tail_end': 10, 'tail_lower_midpont': 11, 'tail_midpoint': 12, 'tail_upper_midpont': 13}
    for i, (x, y) in enumerate(keypoints):
        element = element_base.copy()
        element["points"] = [x, y]
        for label, idx in labels.items():
            if idx == i:
                element["label"] = label
                break
        skeleton_base["elements"].append(element)
    annotation_base[0]["shapes"].append(skeleton_base)
    with open(os.path.join(output_folder, 'annotations.json'), "w") as f:
        json.dump(annotation_base, f)
    if index == 0:
        with open(os.path.join(data_folder, 'manifest.jsonl'), "w") as f:
            f.write('{"version":"1.1"}\n')
            f.write('{"type":"images"}\n')
    with open(os.path.join(data_folder, 'manifest.jsonl'), "a") as f:
        f.write(f"{{\"name\":\"{basename}\",\"extension\":\".png\",\"width\":1920,\"height\":1200,\"meta\":{{\"related_images\":[]}}}}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image files from a folder, list, or single file.")
    parser.add_argument("-i", "--input", required=True, help="Input folder, list of image files (comma-separated), or single image file")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search directories recursively for video files if input is a directory.",
    )

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    filenames = get_image_files(args.input, args.recursive)
    process_name = f"prediction-{datetime.now().strftime("%y%m%d_%H%M%S")}"
    output_folder = f'out/{process_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    prediction_folder = os.path.join(output_folder, 'predictions')
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    data_folder = os.path.join(output_folder, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    model_detection = YOLO(r"out\1k-detection.pt")  # pretrained YOLO11n model

    with open(r'marco\config\hrnet_w32_384_288.yaml', 'r') as f:
        cfg_w32_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w32_384_288['MODEL']['NUM_JOINTS'] = 14
        image_size = cfg_w32_384_288['MODEL']['IMAGE_SIZE']
        model_pose = hrnet.get_pose_net(cfg_w32_384_288, is_train=False)
        model_pose = model_pose.to(device)
        model_pose.load_state_dict(torch.load(r'out\snapshot_PoseHRNet-W32_288x384.pth', weights_only=True, map_location=device))


        # model_pose = SHG.get_pose_net()
        # model_pose = model_pose.to(device)
        # model_pose.load_state_dict(torch.load(r'out\train-SHG-best\snapshot_best.pth', weights_only=True, map_location=device))
        # image_size = [256, 256]

        for idx, filename in enumerate(filenames):
            
            # copy the files into the data folder for the CVAT format
            basename = os.path.basename(filename)
            shutil.copy(filename, data_folder)

            # take bbox information from the object detection and crop the image
            bbox = model_detection(filename)[0].boxes.xyxy[0].to('cpu')
            image = Image.open(filename).convert("RGB")
            image = image.crop(bbox.numpy())
            padding_width, padding_height = calculate_padding(*image.size, *image_size)
            image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(image)
            scale_x = image_size[1] / image.size[0]
            scale_y = image_size[0] / image.size[1]
            image = F.resize(image, image_size)
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.5] * 3, std=[0.5] * 3)
            image = image.unsqueeze(0).to(device)
            output = model_pose(image)
            predictions = output.squeeze(0)
            image = image.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format
            confidence = 0
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            kps = torch.tensor([t[0] for t in keypoints if t[1] > confidence], dtype=torch.float32)
            # Convert keypoints to NumPy and add bounding box offsets
            keypoints_np = (kps.numpy() / np.array([scale_x, scale_y])) + \
                            np.array([bbox[0], bbox[1]]) - \
                            np.array([padding_width, padding_height])
            add_annotation_CVAT(idx, filename, keypoints_np, bbox, output_folder, data_folder)
            image = cv2.imread(filename)
            for x, y in keypoints_np.astype(int):
                cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            # Display image
            # cv2.imshow("Image with Keypoints", image)
            cv2.imwrite(os.path.join(prediction_folder, os.path.basename(filename)), image)
            # cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def update_stop_frame(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "stop_frame":
                    obj[key] = len(filenames) - 1
                else:
                    update_stop_frame(value)
        elif isinstance(obj, list):
            for item in obj:
                update_stop_frame(item)

    with open(r"marco\config\task_base.json", "r") as f:
        task_base = json.load(f)
        task_base["name"] = process_name
        update_stop_frame(task_base)
        with open(os.path.join(output_folder, 'task.json'), 'w', encoding='utf-8') as f2:
            json.dump(task_base, f2, indent=4)

    zip_path = os.path.join(output_folder, process_name + '.zip')
    create_zip(zip_path, data_folder, [os.path.join(output_folder, 'task.json'), os.path.join(output_folder, 'annotations.json')])
