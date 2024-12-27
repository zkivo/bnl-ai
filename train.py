import time
import torch
import signal
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
from torch.utils.data import DataLoader, Dataset, random_split
from hrnet_w32_256 import get_pose_net 
from MouseDataset import *
import os
from datetime import datetime

if platform.system() == "Darwin":  # "Darwin" is the name for macOS
    multiprocessing.set_start_method("fork", force=True)

# signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
# signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals


output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

learning_rate = 0.001
batch_size = 1 
epochs = 100

dataset = MouseDataset(image_folder='data/dataset', 
                       label_file='data/dataset/labels.csv', 
                       output_size=(256, 192),
                       plot=False)

# Define the split ratio
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into training and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=1, num_workers=2, shuffle=False)

model = get_pose_net(is_train=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    for batch_idx, (images, heatmaps) in enumerate(train_dataloader):
        optimizer.zero_grad()
        prediction = model(images)

        loss = criterion(prediction, heatmaps)
        epoch_loss += loss.item()
        loss.backward()

        optimizer.step()
        print(f'[{(batch_idx + 1) * batch_size} / {len(train_dataset)}]: loss: {loss.item()}')

    print(f"Epoch [{epoch}/{epochs}], Average Train Loss: {epoch_loss / len(train_dataset):.4f}" \
          " Time: {:.2f}".format(time.time() - start_time))
    
    model.eval()
    print('Testing...')
    test_loss = 0.0
    for batch_idx, (images, heatmaps) in enumerate(test_dataloader):
        prediction = model(images)
        loss = criterion(prediction, heatmaps)
        test_loss += loss.item()
        # print(f'{batch_idx}: loss: {loss.item()}')

    print(f"Average Test Loss: {test_loss / len(test_dataset):.4f}")

    if epoch % 10 == 0:
        print(f'Saving snapshot at epoch {epoch}...')
        torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{epoch}.pth'))
