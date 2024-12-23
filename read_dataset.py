import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import math

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
        # Calculate bounding box around keypoints
        keypoints_2d = keypoints.view(-1, 2)

        # Filter out invalid (NaN) keypoints
        valid_keypoints = keypoints_2d[~torch.isnan(keypoints_2d).any(dim=1)]
        if len(valid_keypoints) == 0:
            raise ValueError("All keypoints are NaN for this sample.")

        min_x, _ = torch.min(valid_keypoints[:, 0], 0)
        max_x, _ = torch.max(valid_keypoints[:, 0], 0)
        min_y, _ = torch.min(valid_keypoints[:, 1], 0)
        max_y, _ = torch.max(valid_keypoints[:, 1], 0)

        print(valid_keypoints)

        # Add padding as 5% of the bounding box dimensions
        padding_x = 0.20 * (max_x - min_x)
        padding_y = 0.20 * (max_y - min_y)

        min_x = max(0, int(min_x - padding_x))
        min_y = max(0, int(min_y - padding_y))
        max_x = min(image.size[0], int(max_x + padding_x))
        max_y = min(image.size[1], int(max_y + padding_y))

        # Crop the image to the bounding box
        image = image.crop((min_x, min_y, max_x, max_y))

        # Adjust keypoints for cropping
        keypoints_2d[:, 0] -= min_x
        keypoints_2d[:, 1] -= min_y
        keypoints = keypoints_2d.view(-1)

        # Random rotation
        angle = random.uniform(-self.rotation, self.rotation)

        # Calculate bounding box of the rotated image
        w, h = image.size
        angle_rad = math.radians(angle)
        image = F.rotate(image, angle, expand=True)
        new_size = image.size

        # Adjust keypoints for rotation
        center_x, center_y = new_size[0] / 2, new_size[1] / 2
        rotation_matrix = torch.tensor([
            [math.cos(-angle_rad), -math.sin(-angle_rad)],
            [math.sin(-angle_rad), math.cos(-angle_rad)]
        ])
        keypoints = keypoints.view(-1, 2)
        keypoints += torch.tensor([(new_size[0] - w) / 2, (new_size[1] - h) / 2])  # Adjust for padding
        keypoints -= torch.tensor([center_x, center_y])
        keypoints = torch.mm(keypoints, rotation_matrix.T) + torch.tensor([center_x, center_y])
        keypoints = keypoints.view(-1)

        # Resize image to output size
        scale_x = self.output_size[1] / new_size[0]
        scale_y = self.output_size[0] / new_size[1]
        image = F.resize(image, self.output_size)

        # Scale keypoints
        keypoints[::2] *= scale_x  # Scale x-coordinates
        keypoints[1::2] *= scale_y  # Scale y-coordinates

        # Normalize image
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[self.mean] * 3, std=[self.std] * 3)

        return image, keypoints

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
            image, keypoints = self.transform(image, torch.tensor(keypoints))

        return image, keypoints

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
for i, (images, keypoints) in enumerate(data_loader):
    if i % 10 == 0:  # Show every 10th batch
        # Assuming batch size is > 1, take the first image and its keypoints
        image = images[0]  # Tensor of shape (C, H, W)
        keypoint = keypoints[0]  # Corresponding keypoints for the image

        # Convert the image tensor to a numpy array for display
        image_np = F.to_pil_image(image)

        # Plot the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np)
        plt.scatter(keypoint[::2], keypoint[1::2], c='red', s=20)  # Plot keypoints
        plt.title(f"Image {i * len(images)} with Keypoints")
        plt.axis("off")
        plt.show()
