import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
from torch.nn import functional
import matplotlib.pyplot as plt
import torchvision
import math
import cv2
import random
import numpy as np

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

def get_left_right_swap_indices(keypoint_dict):
    """Returns a list of pairs (i, j) where i and j are left/right keypoint indices."""
    swap_indices = []
    names = list(keypoint_dict.keys())
    
    for i, name in enumerate(names):
        if name.startswith('left_'):
            right_name = name.replace('left_', 'right_')
            if right_name in keypoint_dict:
                j = keypoint_dict[right_name]
                swap_indices.append((i, j))
    return swap_indices

class PoseDataset(Dataset):
    def __init__(self, image_folder, resize_to, heatmap_size, label_file=None, augmentation=False):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            label_file (str): Path to the CSV file containing labels (keypoints).
            resize_to (tuple): Imaga size to which the transformation will resize to (height, width).
            rotate (bool): Whether to randomly rotate the images of 90, 180 or 270 degrees.
        """
        self.image_folder = image_folder
        self.resize_to = resize_to
        self.heatmap_size = heatmap_size
        self.labels = pd.read_csv(label_file)
        self.image_names = os.listdir(image_folder)
        self.image_names.sort()
        self.augmentation = augmentation
        self.images = []

        # add to memory all the images
        for image_name in self.image_names:
            img_path = os.path.join(self.image_folder, image_name)
            image = torchvision.io.decode_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # (C, H, W)
            self.images.append(image)

        # bbox
        df = self.labels.copy()
        self.bbox = df[[col for col in df.columns if col.startswith("bbox_")]]
        
        # keypoints
        df = df.drop(columns=[col for col in df.columns if col.startswith("bbox_")])
        keypoints = sorted(set(col.rsplit('-', 1)[0] for col in df.columns if '-' in col))
        ordered_columns = [col for pair in keypoints for col in (f"{pair}-x", f"{pair}-y") if col in df.columns]
        ordered_columns = ['filename'] + ordered_columns if 'filename' in df.columns else ordered_columns
        self.keypoints = df[ordered_columns]

        # keypoint names 
        df_cp = df.copy()
        df_cp = df_cp.drop(columns=['filename'])
        self.keypoint_names = {}
        for idx, col in enumerate(df_cp.columns):
            # Remove '-x' or '-y' from the column name
            base_name = col.rstrip('-xy')
            if base_name not in self.keypoint_names:
                self.keypoint_names[base_name] = idx
        self.swap_keypoints = get_left_right_swap_indices(self.keypoint_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        # ------- steps --------
        # Crop image around mouse
        # rotate with extension = True
        # add padding so that matches next resizing aspect ratio
        # resize to match input model
        # normalize image
        # generate heatmap for each keypoint
        # ----------------------

        transformed_image = self.images[idx].clone()
        img_name = self.image_names[idx]
        idx = self.labels[self.labels['filename'] == img_name].index[0]
        bbox = self.bbox.iloc[idx,:].to_numpy()
        keypoints = self.keypoints[self.keypoints.iloc[:, 0] == img_name].iloc[:, 1:].values.astype('float32') # Rest are keypoints
        keypoints = torch.tensor(keypoints)
        keypoints = keypoints.view(-1, 2)

        # ----------------------------------
        # --- Crop images and keypoints ----
        # ----------------------------------

        # if not self.infer:
        # Filter out invalid (NaN) keypoints
        valid_keypoints = keypoints[~torch.isnan(keypoints).any(dim=1)]
        if len(valid_keypoints) == 0:
            raise ValueError("All keypoints are NaN for this sample.")
        
        transformed_image = F.crop(transformed_image, int(bbox[1]), int(bbox[0]), 
                                   int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]))

        # Crop keypoints
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 1] -= bbox[1]
        keypoints = keypoints.view(-1)

        # ----------------------------------
        # --- Scaling images and keypoints ----
        # ----------------------------------
        if self.augmentation:
            scale = random.uniform(0.9, 1.1)  # Random scale factor between 0.9 and 1.1
            keypoints *= scale
            transformed_image = transformed_image.unsqueeze(0)
            transformed_image = functional.interpolate(transformed_image, 
                                                       scale_factor=scale, 
                                                       mode='bilinear', 
                                                       align_corners=False)
            transformed_image = transformed_image.squeeze(0)


        # ----------------------------------
        # --- Rotate images and keypoints --
        # ----------------------------------

        keypoints = keypoints.view(-1, 2)
        # Random rotation
        if self.augmentation:
            angle = random.choice([0, 90, 180, 270])
            # Calculate bounding box of the rotated image and rotate image
            crop_width, crop_height = transformed_image.shape[2], transformed_image.shape[1]
            angle_rad = math.radians(angle)
            transformed_image = F.rotate(transformed_image, angle, expand=True)
            # Rotate keypoints
            center_x, center_y = transformed_image.shape[2] / 2, transformed_image.shape[1] / 2
            rotation_matrix = torch.tensor([
                [math.cos(-angle_rad), -math.sin(-angle_rad)],
                [math.sin(-angle_rad), math.cos(-angle_rad)]
            ])
            keypoints += torch.tensor([(transformed_image.shape[2] - crop_width) / 2, (transformed_image.shape[1] - crop_height) / 2])  # Adjust for padding
            keypoints -= torch.tensor([center_x, center_y])
            keypoints = torch.mm(keypoints, rotation_matrix.T) + torch.tensor([center_x, center_y])
        

        # ----------------------------------
        # --- Add padding and resize -------
        # ----------------------------------

        # Calculate padding to match the aspect ratio
        padding_width, padding_height = calculate_padding(transformed_image.shape[2], transformed_image.shape[1], *self.resize_to)
        # print(f"padding width: {padding_width}, padding height: {padding_height}")
        transformed_image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(transformed_image)
        # Add padding to keypoints
        keypoints += torch.tensor([padding_width, padding_height])  # Adjust for padding
        keypoints = keypoints.view(-1)
        # Resize image to output size
        scale_x = self.resize_to[1] / transformed_image.shape[2]
        scale_y = self.resize_to[0] / transformed_image.shape[1]
        transformed_image = F.resize(transformed_image, self.resize_to)
        # Resize keypoints
        keypoints[::2] *= scale_x  # Scale x-coordinates
        keypoints[1::2] *= scale_y  # Scale y-coordinates
        padding_width_hm  = int( padding_width * scale_x)
        padding_height_hm = int( padding_height * scale_y)

        # Normalize image
        transformed_image = transformed_image.float() / 255.0

        if self.augmentation:
            _, H, W = transformed_image.shape

            hflip = random.choice([False, True]) # Randomly choose flip mode: 0 = none, 1 = h-flip
            keypoints = keypoints.view(-1, 2)
            if hflip:
                # Horizontal flip
                transformed_image = torch.flip(transformed_image, dims=[2])
                keypoints[:, 0] = W - keypoints[:, 0]
                for i, j in self.swap_keypoints:
                    keypoints[[i, j]] = keypoints[[j, i]]

            color_augment = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
                T.RandomGrayscale(p=0.2),
            ])
            transformed_image = color_augment(transformed_image)

        transformed_image = F.normalize(transformed_image, mean=[0.5] * 3, std=[0.5] * 3)

        # ----------------------------------
        # ------- Generate heatmaps --------
        # ----------------------------------
        heatmaps = []
        keypoints = keypoints.view(-1, 2)
        for keypoint in keypoints:
            heatmap = generate_heatmap(transformed_image, keypoint, 
                                       padding_width=padding_width_hm,
                                       padding_height=padding_height_hm,
                                       heatmap_size=self.heatmap_size)
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)

        return transformed_image, keypoints, heatmaps, torch.tensor([scale_x, scale_y])

def generate_heatmap(image, keypoint, padding_width, padding_height, heatmap_size=(64, 48)):
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

    # Check for NaN values in keypoint
    if torch.isnan(keypoint).any():
        return torch.zeros(heatmap_h, heatmap_w, dtype=torch.float32)

    # Convert keypoint to heatmap space
    x, y = keypoint
    scale_x = heatmap_w / img_w
    scale_y = heatmap_h / img_h
    keypoint_hm = torch.tensor([x * scale_x, y * scale_y])
    padding_width  = int(padding_width * scale_x)
    padding_height = int(padding_height * scale_y)

    def sigma(H, W, base_H=64, base_W=48, base_sigma=1.1):
        base_diag = math.sqrt(base_H ** 2 + base_W ** 2)
        diag = math.sqrt(H ** 2 + W ** 2)
        return base_sigma * (diag / base_diag)

    # Create the Gaussian heatmap
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    center_x, center_y = int(keypoint_hm[0]), int(keypoint_hm[1])

    for i in range(heatmap_h):
        for j in range(heatmap_w):
            heatmap[i, j] = np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma(heatmap_h, heatmap_w) ** 2))

    if padding_height != 0:
        heatmap[:padding_height, :]  = 0  # Top padding
        heatmap[-padding_height:, :] = 0  # Bottom padding
    if padding_width != 0:
        heatmap[:, :padding_width]   = 0  # Left padding
        heatmap[:, -padding_width:]  = 0  # Right padding

    # heatmap += 1e-10

    # Normalize heatmap to range [0, 1]
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # Convert to tensor
    return torch.tensor(heatmap, dtype=torch.float32)

if __name__ == "__main__":
    # Set paths
    image_folder = r"C:\Users\marco\git\PoseEstimation_Mouse\marco\data\aug_images"
    label_file   = r"marco\datasets\Top1k\annotations.csv"
    rows = 1
    cols = 5

    # Create dataset and data loader
    # random_numbers = random.sample(range(len(os.listdir(image_folder))), rows * cols)
    random_numbers = None

    counter = 0
    dataset = PoseDataset(image_folder=image_folder, 
                          label_file=label_file,
                          resize_to=(192, 256),
                          heatmap_size=(48, 64),
                          augmentation=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Iterate through the data loader
    fig, ax = plt.subplots(rows, cols)
    fig.set_size_inches(10, 9)
    for i, (images, gt_kps, gt_hms, _) in enumerate(data_loader):
        # if i >= rows * cols:  # Show every 10th batch
        #     plt.show()
        #     break
        if random_numbers:
            if i not in random_numbers:
                continue
        image = images[0]
        gt_kps = gt_kps[0]

        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
        denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
        denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
        denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
        denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

        gt_hms = gt_hms[0]
        # fig, ax = plt.subplots(7, 2)
        # for i in range(7):
        #     ax[i, 0].imshow(gt_hms[i], cmap='hot', interpolation='nearest')
        #     ax[i, 0].axis('off')
        #     ax[i, 1].imshow(gt_hms[i + 7], cmap='hot', interpolation='nearest')
        #     ax[i, 1].axis('off')
        # plt.show()        
        # fig2, ax2 = plt.subplots(1, 2)
        # ax2[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax2[0].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # ax2[1].imshow(denormalized_image)
        # ax2[1].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # plt.show()

        overlap_hm = gt_hms[0]
        for hm in gt_hms[1:]:
            try:
                overlap_hm = torch.maximum(overlap_hm, hm)        
            except Exception as e: 
                if overlap_hm is None:
                    overlap_hm = hm
        # ax[int(counter / cols), counter % cols].imshow(denormalized_image)
        ax[counter % cols].imshow(denormalized_image)
        ax[counter % cols].axis('off')
        # ax[int(counter / 5), counter % 5].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # ax[1, ].imshow(overlap_hm, cmap='hot', interpolation='nearest')
        # ax[0, ].imshow(denormalized_image)
        # ax[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax[1].scatter(keypoints[:,0], keypoints[:,1], c='red', s=20)  # Plot keypoints
        # ax[0].imshow(cropped_image[0])
        counter = counter + 1

    counter = 0
    dataset = PoseDataset(image_folder=image_folder, 
                          label_file=label_file,
                          resize_to=(192, 256),
                          heatmap_size=(48, 64),
                          augmentation=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    fig, ax = plt.subplots(rows, cols)
    fig.set_size_inches(10, 9)
    for i, (images, gt_kps, gt_hms, _) in enumerate(data_loader):
        # if i >= rows * cols:  # Show every 10th batch
        #     plt.show()
        #     break
        if random_numbers:
            if i not in random_numbers and random_numbers is not None:
                continue
        image = images[0]
        gt_kps = gt_kps[0]

        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
        denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
        denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
        denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
        denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

        gt_hms = gt_hms[0]
        # fig, ax = plt.subplots(7, 2)
        # for i in range(7):
        #     ax[i, 0].imshow(gt_hms[i], cmap='hot', interpolation='nearest')
        #     ax[i, 0].axis('off')
        #     ax[i, 1].imshow(gt_hms[i + 7], cmap='hot', interpolation='nearest')
        #     ax[i, 1].axis('off')
        # plt.show()        
        # fig2, ax2 = plt.subplots(1, 2)
        # ax2[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax2[0].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # ax2[1].imshow(denormalized_image)
        # ax2[1].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # plt.show()

        overlap_hm = gt_hms[0]
        for hm in gt_hms[1:]:
            try:
                overlap_hm = torch.maximum(overlap_hm, hm)        
            except Exception as e: 
                if overlap_hm is None:
                    overlap_hm = hm
        ax[counter % cols].imshow(denormalized_image)
        ax[counter % cols].axis('off')
        # ax[int(counter / 5), counter % 5].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # ax[1, ].imshow(overlap_hm, cmap='hot', interpolation='nearest')
        # ax[0, ].imshow(denormalized_image)
        # ax[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax[1].scatter(keypoints[:,0], keypoints[:,1], c='red', s=20)  # Plot keypoints
        # ax[0].imshow(cropped_image[0])
        counter = counter + 1
    
    plt.show()

    # dataset = PoseDataset(image_folder=image_folder, 
    #                       label_file=label_file,
    #                       resize_to=(192, 256),
    #                       heatmap_size=(48, 64),
    #                     #   resize_to=(288, 384),
    #                     #   heatmap_size=(72, 94),
    #                       augmentation=False)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # # fig = plt.figure(figsize=(15, 10))
    # for i, (images, gt_kps, gt_hms, _) in enumerate(data_loader):
    #     if i >= 10:  # Show every 10th batch
    #         plt.show()
    #         break
    #     if random_numbers:
    #         if i not in random_numbers and random_numbers is not None:
    #             continue
    #     image = images[0]
    #     gt_kps = gt_kps[0]

    #     mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
    #     std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to('cpu')
    #     denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
    #     denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
    #     denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
    #     denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

        # gt_hms = gt_hms[0]
        # fig, ax = plt.subplots(7, 2)
        # for i in range(7):
        #     ax[i, 0].imshow(gt_hms[i], cmap='plasma', interpolation='nearest')
        #     ax[i, 0].axis('on')
        #     ax[i, 1].imshow(gt_hms[i + 7], cmap='plasma', interpolation='nearest')
        #     ax[i, 1].axis('on')
        # fig2, ax2 = plt.subplots(1, 2)
        # ax2[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax2[0].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # ax2[1].imshow(denormalized_image)
        # ax2[1].scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # plt.show()

        # overlap_hm = gt_hms[0]
        # for hm in gt_hms[1:]:
        #     try:
        #         overlap_hm = torch.maximum(overlap_hm, hm)        
        #     except Exception as e: 
        #         if overlap_hm is None:
        #             overlap_hm = hm
        # plt.imshow(denormalized_image)
        # plt.scatter(gt_kps[:,0], gt_kps[:,1], c='red', s=20)  # Plot keypoints
        # # print names of keypoints
        # for name, idx in dataset.keypoint_names.items():
        #     plt.text(gt_kps[idx][0], gt_kps[idx][1], name, fontsize=12, color='white')
        # ax[1, ].imshow(overlap_hm, cmap='plasma', interpolation='nearest')
        # ax[0, ].imshow(denormalized_image)
        # ax[0].imshow(image.numpy().transpose(1, 2, 0))
        # ax[1].scatter(keypoints[:,0], keypoints[:,1], c='red', s=20)  # Plot keypoints
        # ax[0].imshow(cropped_image[0])
        # plt.show()

