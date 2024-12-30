import csv
import time
import torch
import signal
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from hrnet_w32_256 import get_pose_net, PoseHighResolutionNet
from TopViewDataset import TopViewDataset
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import os
from datetime import datetime

if platform.system() == "Darwin":  # "Darwin" is the name for macOS
    multiprocessing.set_start_method("fork", force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Torch version: {torch.__version__}')
print(f'Torch device: {device}')

# if device.type == 'cpu':
#     torch.set_num_threads(torch.get_num_threads())
#     torch.set_num_interop_threads(torch.get_num_interop_threads())

# def signal_handler(sig, frame):
#     print('Saving loss graph...')
#     plt.figure()
#     plt.plot(train_losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Train Loss')
#     plt.savefig(os.path.join(output_folder, 'train_loss.png'))
#     exit(0)

# signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
# signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals


output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset = TopViewDataset(image_folder='data/dataset', 
                       label_file='data/dataset/labels.csv', 
                       output_size=(256, 192),
                       debug=True)

dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

model = PoseHighResolutionNet().to(device)
model.load_state_dict(torch.load('out/train-241229_132945/snapshot_100.pth', weights_only=True, map_location=torch.device('cpu')))
model.eval()
criterion = nn.MSELoss()

def overlay_keypoints_with_confidence(image, keypoints_with_confidence):
    """
    Overlays keypoints with confidence on an image.
    Args:
        image: Input image as a NumPy array.
        keypoints_with_confidence: List of ((x, y), confidence) for each keypoint.
    """
    plt.imshow(image)
    for (y, x), confidence in keypoints_with_confidence:
        if confidence < 0.5:
            continue
        plt.scatter(x, y, c='red', s=30)  # Plot keypoints as red dots
        plt.text(x, y - 5, f"{confidence*100:.1f}%", color='yellow', fontsize=8, ha='center')
    plt.axis('off')


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
        keypoints_with_confidence.append(((y, x), confidence))
    return keypoints_with_confidence


with torch.no_grad():
    test_loss = 0.0
    num_batches = 0
    for batch_idx, (images, gt_hm, original_image, not_normalized_image) in enumerate(dataloader):
        num_batches += 1
        predictions = model(images)

        for img_idx, (image, heatmaps) in enumerate(zip(images, predictions)):
            # Rescale heatmaps to match image size
            num_keypoints, h, w = heatmaps.shape
            original_size = image.shape[1:]  # Assume CHW format

            # resized_heatmaps = torch.stack([F.interpolate(hm.unsqueeze(0), size=(256, 192), mode="bilinear", align_corners=False).squeeze(0) for hm in heatmaps])
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size, ) for hm in heatmaps])
            # Extract keypoints from heatmaps
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            # Convert image to NumPy for plotting
            image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC

            # Display image with keypoints
            plt.figure(figsize=(8, 8))
            plt.title("Image with Keypoints")
            overlay_keypoints_with_confidence(not_normalized_image.squeeze(0), keypoints)
            plt.show()
            # Display all heatmaps
            # plt.figure(figsize=(15, 10))
            # for i, heatmap in enumerate(resized_heatmaps):
            #     plt.subplot(2, int(np.ceil(num_keypoints / 2)), i + 1)
            #     plt.imshow(heatmap.cpu().numpy(), cmap='hot')
            #     plt.title(f"Keypoint {i + 1}")
            #     plt.axis('off')
            # plt.suptitle("Heatmaps")
            # plt.show()

        # loss = criterion(predictions, heatmaps)
        # test_loss += loss.item()
        # print(f'{batch_idx}: loss: {loss.item()}')
    # test_loss /= num_batches
