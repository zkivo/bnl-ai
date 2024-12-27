import time
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
from torch.utils.data import DataLoader, Dataset, random_split
from hrnet_w32_256 import get_pose_net 
from data_loader import *
import os
from datetime import datetime


if platform.system() == "Darwin":  # "Darwin" is the name for macOS
    multiprocessing.set_start_method("fork", force=True)

output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

learning_rate = 0.001
batch_size = 1 
epochs = 100

transform = ImageKeypointTransform(output_size=(256, 192), mean=0.5, std=0.5)
dataset = MouseDataset(image_folder='data/dataset', label_file='data/dataset/labels.csv', transform=transform)

# Define the split ratio
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into training and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=1,  num_workers=2, shuffle=False)

model = get_pose_net(is_train=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    for batch_idx, (images, cropped_image, keypoints, padding_width, padding_height) in enumerate(train_dataloader):
        heatmaps = []
        keypoints = keypoints.view(-1, 2)
        for i, keypoint in enumerate(keypoints):
            heatmap = generate_heatmap(images[0], keypoint, padding_width=padding_width,
                                        padding_height=padding_height,
                                        heatmap_size=(64, 48), sigma=2)
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)
        optimizer.zero_grad()
        # Forward pass
        prediction = model(images)
        loss = criterion(prediction, heatmaps)
        epoch_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'{batch_idx}: loss: {loss.item()}')

    print(f"Epoch [{epoch + 1}/{epochs}], Average Train Loss: {epoch_loss / len(train_dataset):.4f}" \
          " Time: {:.2f}".format(time.time() - start_time))
    
    model.eval()
    print('Testing...')
    test_loss = 0.0
    for batch_idx, (images, cropped_image, keypoints, padding_width, padding_height) in enumerate(test_dataloader):
        heatmaps = []
        keypoints = keypoints.view(-1, 2)
        for i, keypoint in enumerate(keypoints):
            heatmap = generate_heatmap(images[0], keypoint, padding_width=padding_width,
                                        padding_height=padding_height,
                                        heatmap_size=(64, 48), sigma=2)
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)
        # Forward pass
        prediction = model(images)

        # Compute loss
        loss = criterion(prediction, heatmaps)
        test_loss += loss.item()

        # print(f'{batch_idx}: loss: {loss.item()}')

    print(f"Average Test Loss: {test_loss / len(test_dataset):.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{epoch}.pth'))
