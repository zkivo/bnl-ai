from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
import hrnet
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import yaml
import math
import cv2
import csv
import sys
import os

def calculate_padding(original_width, original_height, target_width, target_height):
    target_height, target_width = target_width, target_height
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

def add_margin_bbox(bbox, margin=0.05):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    pad_w = width * margin
    pad_h = height * margin

    new_x1 = x1 - pad_w
    new_y1 = y1 - pad_h
    new_x2 = x2 + pad_w
    new_y2 = y2 + pad_h

    return [new_x1, new_y1, new_x2, new_y2]

def transform_frame(frame, target_width, target_height, bbox_tl_x, bbox_tl_y, bbox_br_x, bbox_br_y):
    transformed_image = frame.copy()
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    transformed_image = transformed_image.astype(np.float32) / 255.0
    transformed_image = torch.from_numpy(transformed_image).permute(2, 0, 1)

    # crop image
    transformed_image = F.crop(transformed_image, int(bbox_tl_y), int(bbox_tl_x), 
                               int(bbox_br_y - bbox_tl_y), int(bbox_br_x - bbox_tl_x))
    
    # padding and resize to input of model
    padding_width, padding_height = calculate_padding(transformed_image.shape[2], transformed_image.shape[1], target_width, target_height)
    transformed_image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(transformed_image)
    transformed_image = F.resize(transformed_image, (target_width, target_height))

    # normalize
    transformed_image = F.normalize(transformed_image, mean=[0.5] * 3, std=[0.5] * 3)
    
    # plt.imshow(transformed_image.numpy().transpose(1, 2, 0))
    # plt.show()

    return transformed_image, padding_width, padding_height

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

def infer_video(labels, n_joints=14, plot=True):
    if len(sys.argv) < 2:
        print("Usage: python infer_video.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        sys.exit(1)

    base, ext = os.path.splitext(video_path)
    inference_video_path = f"{base}-inference{ext}"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(inference_video_path, fourcc, fps, (frame_width, frame_height))

    csv_inference = f"{base}-inference{'.csv'}"
    with open(csv_inference, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_number', 'back_croup-x', 'back_midpoint-x', 'back_withers-x', 'ears_midpoint-x', 'left_ear_base-x', 'left_ear_tip-x', 'nose-x', 'right_ear_base-x', 'right_ear_tip-x', 'tail_base-x', 'tail_end-x', 'tail_lower_midpoint-x', 'tail_midpoint-x', 'tail_upper_midpoint-x', 'back_croup-y', 'back_midpoint-y', 'back_withers-y', 'ears_midpoint-y', 'left_ear_base-y', 'left_ear_tip-y', 'nose-y', 'right_ear_base-y', 'right_ear_tip-y', 'tail_base-y', 'tail_end-y', 'tail_lower_midpoint-y', 'tail_midpoint-y', 'tail_upper_midpoint-y', 'bbox_tl-x', 'bbox_tl-y', 'bbox_br-x', 'bbox_br-y'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    with open(r'marco\config\hrnet_w48_256_192.yaml', 'r') as f:
        yaml_text = f.read()
    cfg_w48_256_192 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    cfg_w48_256_192['MODEL']['NUM_JOINTS'] = n_joints
    input_size  = cfg_w48_256_192['MODEL']['IMAGE_SIZE']
    output_size = cfg_w48_256_192['MODEL']['HEATMAP_SIZE']
    
    model_pose = hrnet.get_pose_net(cfg_w48_256_192, is_train=False)
    model_pose = model_pose.to(device)
    model_pose.load_state_dict(torch.load(r"trained_models\pose_RQ3\snapshot_PoseHRNet-W48_192x256.pth", weights_only=True, map_location=device))
    model_detect = YOLO(r"trained_models\detection_Top_1k_300epochs\weights\best.pt")  # pretrained YOLO11n model

    confidence = 0.6
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # original_height, original_width, _ = frame.shape
        result = model_detect(frame, stream=False, verbose=False)[0]
        bbox = result.boxes.xyxy[0].cpu().numpy()
        bbox = add_margin_bbox(bbox, margin=0.03)
        transformed_image, padding_width, padding_height = transform_frame(frame, *input_size, *bbox)
        transformed_image = transformed_image.unsqueeze(0).to(device)
        predictions = model_pose(transformed_image)
        transformed_image = transformed_image.squeeze(0)
        predictions = predictions.squeeze(0)
        resized_heatmaps = torch.stack([F.resize(hm.unsqueeze(0), (input_size[1], input_size[0])) for hm in predictions])
        keypoints_and_confidance = extract_keypoints_with_confidence(resized_heatmaps)
        keypoints = torch.tensor([k if c > confidence else (float('Nan'), float('Nan'))  for k, c in keypoints_and_confidance], dtype=torch.float32)
        scale_x = input_size[0] / (bbox[2] - bbox[0] + 2 * padding_width) # transformed_image.shape[2]
        scale_y = input_size[1] / (bbox[3] - bbox[1] + 2 * padding_height) # transformed_image.shape[1]
        keypoints_to_original_frame = (keypoints.numpy() / np.array([scale_x, scale_y])) + \
                                       np.array([bbox[0], bbox[1]]) - \
                                       np.array([padding_width, padding_height])
        for x, y in keypoints_to_original_frame.astype(float):
            if not math.isnan(x) and not math.isnan(y):
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        with open(csv_inference, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([frame_idx, *keypoints_to_original_frame[:, 0], *keypoints_to_original_frame[:, 1], *bbox])
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(200,200,200), thickness=2)
        if plot:
            cv2.imshow('inference', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"frame: {frame_idx}", end="\r", flush=True)
        frame_idx += 1

    cap.release()
    video_writer.release()
    print("Finished inferencing video.")

if __name__ == "__main__":

    labels = {
        0: 'back_croup',
        1: 'back_midpoint',
        2: 'back_withers',
        3: 'ears_midpoint',
        4: 'left_ear_base',
        5: 'left_ear_tip',
        6: 'nose',
        7: 'right_ear_base',
        8: 'right_ear_tip',
        9: 'tail_base',
        10: 'tail_end',
        11: 'tail_lower_midpoint',
        12: 'tail_midpoint',
        13: 'tail_upper_midpoint'
    }

    infer_video(labels, n_joints=14, plot=False)
