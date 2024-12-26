import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import numpy as np

def calculate_padding(original_width, original_height, target_width, target_height):
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

class ImageKeypointTransform:
    def __init__(self, rotation=180, output_size=(256, 192), mean=0.5, std=0.5):
        """
        Args:
            rotation (int): Maximum rotation angle in degrees.
            output_size (tuple): Desired output size (height, width).
            mean (float): Mean for normalization.
            std (float): Standard deviation for normalization.
        """
        self.rotation = rotation
        self.output_size = output_size
        self.mean = mean
        self.std = std

    def __call__(self, image, keypoints):
        # steps:
        # Crop image around mouse
        # rotate with extension = True
        # add padding so that matches next resizing aspect ratio
        # resize to match input model
        # normalize image

        keypoints_2d = keypoints.view(-1, 2)

        # ----------------------------------
        # --- Crop images and keypoints ----
        # ----------------------------------

        # Filter out invalid (NaN) keypoints
        valid_keypoints = keypoints_2d[~torch.isnan(keypoints_2d).any(dim=1)]
        if len(valid_keypoints) == 0:
            raise ValueError("All keypoints are NaN for this sample.")
        min_x, _ = torch.min(valid_keypoints[:, 0], 0)
        max_x, _ = torch.max(valid_keypoints[:, 0], 0)
        min_y, _ = torch.min(valid_keypoints[:, 1], 0)
        max_y, _ = torch.max(valid_keypoints[:, 1], 0)
        # Add padding as 5% of the bounding box dimensions
        padding_x = 0.20 * (max_x - min_x)
        padding_y = 0.20 * (max_y - min_y)
        min_x = max(0, int(min_x - padding_x))
        min_y = max(0, int(min_y - padding_y))
        max_x = min(image.size[0], int(max_x + padding_x))
        max_y = min(image.size[1], int(max_y + padding_y))
        image = image.crop((min_x, min_y, max_x, max_y))
        mean_cropped = np.mean(image)
        std_cropped  = np.std(image)
        # Crop keypoints
        keypoints_2d[:, 0] -= min_x
        keypoints_2d[:, 1] -= min_y
        keypoints = keypoints_2d.view(-1)

        # ----------------------------------
        # --- Rotate images and keypoints --
        # ----------------------------------

        # Random rotation
        # angle = random.uniform(-self.rotation, self.rotation)
        angle = random.choice([90, 180, 270])
        # Calculate bounding box of the rotated image and rotate image
        crop_width, crop_height = image.size
        angle_rad = math.radians(angle)
        image = F.rotate(image, angle, expand=True)
        # Rotate keypoints
        center_x, center_y = image.size[0] / 2, image.size[1] / 2
        rotation_matrix = torch.tensor([
            [math.cos(-angle_rad), -math.sin(-angle_rad)],
            [math.sin(-angle_rad), math.cos(-angle_rad)]
        ])
        keypoints = keypoints.view(-1, 2)
        keypoints += torch.tensor([(image.size[0] - crop_width) / 2, (image.size[1] - crop_height) / 2])  # Adjust for padding
        keypoints -= torch.tensor([center_x, center_y])
        keypoints = torch.mm(keypoints, rotation_matrix.T) + torch.tensor([center_x, center_y])

        # ----------------------------------
        # --- Add padding and resize -------
        # ----------------------------------

        # Calculate padding to match the aspect ratio
        padding_width, padding_height = calculate_padding(*image.size, *self.output_size)
        # print(f"padding width: {padding_width}, padding height: {padding_height}")
        image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(image)
        # Add padding to keypoints
        keypoints += torch.tensor([padding_width, padding_height])  # Adjust for padding
        keypoints = keypoints.view(-1)
        # Resize image to output size
        scale_x = self.output_size[1] / image.size[0]
        scale_y = self.output_size[0] / image.size[1]
        image = F.resize(image, self.output_size)
        # Resize keypoints
        keypoints[::2] *= scale_x  # Scale x-coordinates
        keypoints[1::2] *= scale_y  # Scale y-coordinates
        padding_width  = int( padding_width * scale_x)
        padding_height = int( padding_height * scale_y)
        print(f"padding width: {padding_width}, padding height: {padding_height}")

        # Normalize image
        image = F.to_tensor(image)        
        # image = F.normalize(image, mean=[image.mean(), image.mean(), image.mean()],
        #                      std=[image.std(), image.std(), image.std()])
        image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # print(f'image size: {image.size()}')
        # print(f'image aspect ratio: {image.size(1) / image.size(2)}')
        # print(f'output aspect ratio: {self.output_size[0] / self.output_size[1]}')
        # print(f'mean: {image.mean()}')
        # print(f'std: {image.std()}')
        # print(f'min: {image.min()}')
        # print(f'max: {image.max()}')

        return image, keypoints, padding_width, padding_height

class MousePoseDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            label_file (str): Path to the CSV file containing labels (keypoints).
            transform (callable, optional): Optional transform to be applied on an image and keypoints.
        """
        self.image_folder = image_folder
        self.labels = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image file name and keypoints
        img_name = self.labels.iloc[idx, 0]  # Assuming the first column is the image file name
        img_path = os.path.join(self.image_folder, img_name)
        keypoints = self.labels.iloc[idx, 1:].values.astype('float32')  # Rest are keypoints

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image, keypoints, padding_width, padding_height = self.transform(image, torch.tensor(keypoints))

        return image, keypoints, padding_width, padding_height

def generate_heatmap(image, keypoint, padding_width, padding_height, heatmap_size=(64, 48), sigma=2):
    """
    Generates a heatmap for a given keypoint while handling black padding.

    Args:
        image (torch.Tensor): Original image tensor of shape (C, H, W).
        keypoint (torch.Tensor): Keypoint tensor of shape (2,), in (x, y) format.
        padding_width (int): Width of the padding.
        padding_height (int): Height of the padding.
        heatmap_size (tuple): Output heatmap size (height, width).
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        torch.Tensor: Heatmap tensor of shape (1, height, width).
    """
    # Unpack dimensions
    _, img_h, img_w = image.shape
    heatmap_h, heatmap_w = heatmap_size

    # Convert keypoint to heatmap space
    x, y = keypoint
    scale_x = heatmap_w / img_w
    scale_y = heatmap_h / img_h
    keypoint_hm = torch.tensor([x * scale_x, y * scale_y])
    padding_width  = int(padding_width * scale_x)
    padding_height = int(padding_height * scale_y)

    # Create the Gaussian heatmap
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    center_x, center_y = int(keypoint_hm[0]), int(keypoint_hm[1])

    for i in range(heatmap_h):
        for j in range(heatmap_w):
            heatmap[i, j] = np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))

    if padding_height != 0:
        heatmap[:padding_height, :]  = 0  # Top padding
        heatmap[-padding_height:, :] = 0  # Bottom padding
    if padding_width != 0:
        heatmap[:, :padding_width]   = 0  # Left padding
        heatmap[:, -padding_width:]  = 0  # Right padding

    # Normalize heatmap to range [0, 1]
    heatmap /= heatmap.max()

    # Convert to tensor
    return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

# Set paths
image_folder = "data/dataset"
label_file = "data/dataset/labels.csv"

# Define transformations
transform = ImageKeypointTransform(rotation=180, output_size=(256, 192), mean=0.5, std=0.5)

# Create dataset and data loader
dataset = MousePoseDataset(image_folder=image_folder, label_file=label_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Iterate through the data loader
# selec a random integer between 0 and dataset length
index = random.randint(0, len(dataset))
for i, (images, keypoints, padding_width, padding_height) in enumerate(data_loader):
    if i % 5 == 0:  # Show every 10th batch
        # Assuming batch size is > 1, take the first image and its keypoints
        image = images[0]  # Tensor of shape (C, H, W)
        keypoint = keypoints[0]
        heatmap = generate_heatmap(image, keypoint[0:2], padding_width=padding_width,
                                    padding_height=padding_height,
                                    heatmap_size=(64, 48), sigma=2)

        # Visualize heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap.squeeze(0), cmap='hot', interpolation='nearest')

        # Plot the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.scatter(keypoint[::2], keypoint[1::2], c='red', s=20)  # Plot keypoints
        plt.title(f"Image {i * len(images)} with Keypoints")
        plt.axis("off")
        plt.show()



