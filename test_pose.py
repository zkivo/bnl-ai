import csv
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PoseDataset import PoseDataset
import hrnet
import resnet
import SHG
import pandas as pd
from torchvision.transforms.functional import resize
import os
from datetime import datetime

top_labels = {
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

def test_pose(model, image_test_folder, annotation_path, input_size, output_size, device, output_folder=None, confidence=0):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    coverage = 0.0
    tot_kps = 0
    model.eval()

    if model.name:
        model_name = model.name
    else:
        model_name = model.__class__.__name__
    
    print(f"{model_name}-{input_size}, confidence = {confidence}")

    if output_folder is None:
        output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    image_folder = os.path.join(output_folder, f'test_images_{model_name}_{"x".join(str(n) for n in input_size)}_{confidence}')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    test_dataset  = PoseDataset(image_folder=image_test_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                augmentation=False)
    test_batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=0, shuffle=False)

    # info_file = os.path.join(output_folder, 'info.txt')
    # with open(info_file, 'w') as file:
    #     file.write(f"Model name: {model_name}\n")
    #     file.write(f"number parameters: {sum(p.numel() for p in model.parameters())}\n")
    #     file.write(f"image_test_folder: {image_test_folder}\n")
    #     file.write(f"annotation_path: {annotation_path}\n")
    #     file.write(f"input_size: {input_size}\n")
    #     file.write(f"output_size: {output_size}\n")
    #     file.write(f"test_batch_size: {test_batch_size}\n")
    #     file.write(f"keypoint_names: {test_dataset.keypoint_names}\n")

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

    with torch.no_grad():
        test_loss = 0.0
        pck = []
        list_SE = []
        list_not_nans = []
        count_valid_keypoints = None
        count_covered_keypoints = None
        for batch_idx, (images, gt_keypoints, gt_hms, scaler_kps) in enumerate(dataloader):
            images = images.to(device)
            start = time.time()
            predictions = model(images)
            end = time.time()
            # print(f"Image {batch_idx} took {end - start:.4f} seconds")
            image = images.squeeze(0)
            gt_keypoints = gt_keypoints.squeeze(0)
            gt_hms = gt_hms.squeeze(0)
            predictions = predictions.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format

            # caclulating keypoints from heatmaps
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size, ) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            keypoints_without_confidence = torch.tensor([t[0] for t in keypoints], dtype=torch.float32)
            # coverage += torch.sum(confidence_mask).item()
            # filtered_keypoints = keypoints_without_confidence[confidence_mask]
            # filtered_gt_keypoints = gt_keypoints[confidence_mask]
            tot_kps += gt_keypoints.shape[0] * test_batch_size
            # count gt keypoints, count covered keypoints
            if count_valid_keypoints is None:
                count_valid_keypoints = torch.zeros(gt_keypoints.shape[0], dtype=torch.float32)
            if count_covered_keypoints is None:
                count_covered_keypoints = torch.zeros(gt_keypoints.shape[0], dtype=torch.float32)
            count_valid_keypoints   += (~torch.isnan(gt_keypoints[:, 0])).float()
            count_covered_keypoints += (torch.tensor([t[1] for t in keypoints], dtype=torch.float32) >= confidence).float()
            
            confident_kps = torch.tensor([t[0] if t[1] >= confidence else (float('nan'), float('nan')) for t in keypoints], dtype=torch.float32)
            # confident_gt_kps = torch.tensor([
            #     gt_keypoints[idx].tolist() if keypoints[idx][1] >= confidence else [float('nan'), float('nan')]
            #     for idx in range(len(keypoints))
            # ], dtype=torch.float32)

            list_SE.append(torch.sum((confident_kps/scaler_kps - gt_keypoints/scaler_kps) ** 2, dim=1))
            list_not_nans.append(~torch.isnan(confident_kps[:, 0]) & ~torch.isnan(gt_keypoints[:, 0]))
            
            # Denormalize the image
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

            # Plot the image
            plt.figure(figsize=(7, 7))
            plt.imshow(denormalized_image)
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            plt.scatter(confident_kps[:, 0], confident_kps[:, 1], c='red', marker='x', label="Predicted")  # Predictions in red
            for i, (x, y) in enumerate(confident_kps):
                if np.isnan(x) or np.isnan(y):
                    continue
                plt.text(x + 2, y - 2, top_labels[i], fontsize=8, color='red')
            plt.legend()
            plt.axis("on")
            plt.axis("equal")
            figure_path = os.path.join(image_folder, str(batch_idx))
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            # plt.show()
            # print(f"Image saved at {figure_path}")

        print(f"count_valid_keypoints: {count_valid_keypoints}")
        print(f"count_covered_keypoints: {count_covered_keypoints}")

        stacked_SE = torch.stack(list_SE)
        not_nans = torch.stack(list_not_nans)
        count_not_nans_per_keypoint = torch.count_nonzero(not_nans, dim=(0))
        means_per_keypoint = torch.nansum(stacked_SE, dim=(0)) / count_not_nans_per_keypoint
        RMSE_per_keypoint = torch.sqrt(means_per_keypoint)
        coverage_per_keypoint = count_covered_keypoints / len(dataloader.dataset)
        print(f"RMSE per keypoint: {RMSE_per_keypoint}")
        print(f"Coverage per keypoint: {coverage_per_keypoint}")

        RMSE = torch.sqrt(torch.nanmean(stacked_SE))
        coverage = torch.sum(count_covered_keypoints) / tot_kps
        RMSE = RMSE.item()
        coverage = coverage.item()
        print('RMSE', RMSE)
        print('coverage', coverage)

        return RMSE, RMSE_per_keypoint, coverage, coverage_per_keypoint

if __name__ == '__main__':
    image_test_folder  = r'marco\datasets\Top1k\test'
    annotation_path    = r'marco\datasets\Top1k\annotations.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_folder = os.path.join("out", f'test_pose-{datetime.now().strftime("%y%m%d_%H%M%S")}')

    confidence_range = np.arange(0.0, 0.05, 0.05)

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    # -------------------------------
    # ----------- SHN-2
    # -------------------------------

    # model = SHG.get_pose_net(nstack=2)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ1\snapshot_PoseSHG-2_128x128.pth", weights_only=True, map_location=device))    
    # table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    # for confidence in confidence_range:
    #     rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
    #             test_pose(model, image_test_folder, annotation_path,
    #                       input_size=[128, 128],
    #                       output_size=[32, 32],
    #                       device=device,
    #                       output_folder=output_folder,
    #                       confidence=confidence)
    #     table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
    #                                              "RMSE": rmse, 
    #                                              "coverage": coverage, 
    #                                              "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
    #                                              "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
    #                                              ignore_index=True)
    # table.to_csv(os.path.join(output_folder, 'SHN_2_128x128-RMSEs.csv'), index=False)

    # model = model.to(device)
    # model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ1\snapshot_PoseSHG-2_256x256.pth", weights_only=True, map_location=device))    
    # table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    # for confidence in confidence_range:
    #     rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
    #             test_pose(model, image_test_folder, annotation_path,
    #                       input_size=[256, 256],
    #                       output_size=[64, 64],
    #                       device=device,
    #                       output_folder=output_folder,
    #                       confidence=confidence)
    #     table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
    #                                              "RMSE": rmse, 
    #                                              "coverage": coverage, 
    #                                              "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
    #                                              "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
    #                                              ignore_index=True)
    # table.to_csv(os.path.join(output_folder, 'SHN_2_256x256-RMSEs.csv'), index=False)

    # -------------------------------
    # ----------- HRNet-W32
    # -------------------------------

    with open(r'marco\config\hrnet_w32_256_192.yaml', 'r') as f: yaml_text = f.read()
    cfg_w32_256_192 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 14
    model = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
    model = model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ3\snapshot_PoseHRNet-W32_192x256.pth", weights_only=True, map_location=device))
    table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    for confidence in confidence_range:
        rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
                test_pose(model, image_test_folder, annotation_path,
                          input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                          output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'],
                          device=device,
                          output_folder=output_folder,
                          confidence=confidence)
        table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
                                                 "RMSE": rmse, 
                                                 "coverage": coverage, 
                                                 "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
                                                 "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
                                                 ignore_index=True)
    table.to_csv(os.path.join(output_folder, 'hrnet_w32_256_192-RMSEs.csv'), index=False)

    with open(r'marco\config\hrnet_w32_384_288.yaml', 'r') as f: yaml_text = f.read()
    cfg_w32_384_288 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    cfg_w32_384_288['MODEL']['NUM_JOINTS'] = 14
    model = hrnet.get_pose_net(cfg_w32_384_288, is_train=False)
    model = model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ3\snapshot_PoseHRNet-W32_288x384.pth", weights_only=True, map_location=device))
    table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    for confidence in confidence_range:
        rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
                test_pose(model, image_test_folder, annotation_path,
                          input_size=cfg_w32_384_288['MODEL']['IMAGE_SIZE'],
                          output_size=cfg_w32_384_288['MODEL']['HEATMAP_SIZE'],
                          device=device,
                          output_folder=output_folder,
                          confidence=confidence)
        table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
                                                 "RMSE": rmse, 
                                                 "coverage": coverage, 
                                                 "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
                                                 "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
                                                 ignore_index=True)
    table.to_csv(os.path.join(output_folder, 'hrnet_w32_384_288-RMSEs.csv'), index=False)

    # -------------------------------
    # ----------- HRNet-W48
    # -------------------------------

    with open(r'marco\config\hrnet_w48_256_192.yaml', 'r') as f: yaml_text = f.read()
    cfg_w48_256_192 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    cfg_w48_256_192['MODEL']['NUM_JOINTS'] = 14
    model = hrnet.get_pose_net(cfg_w48_256_192, is_train=False)
    model = model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ3\snapshot_PoseHRNet-W48_192x256.pth", weights_only=True, map_location=device))
    table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    for confidence in confidence_range:
        rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
                test_pose(model, image_test_folder, annotation_path,
                          input_size=cfg_w48_256_192['MODEL']['IMAGE_SIZE'],
                          output_size=cfg_w48_256_192['MODEL']['HEATMAP_SIZE'],
                          device=device,
                          output_folder=output_folder,
                          confidence=confidence)
        table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
                                                 "RMSE": rmse, 
                                                 "coverage": coverage, 
                                                 "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
                                                 "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
                                                 ignore_index=True)
    table.to_csv(os.path.join(output_folder, 'hrnet_w48_256_192-RMSEs.csv'), index=False)

    with open(r'marco\config\hrnet_w48_384_288.yaml', 'r') as f: yaml_text = f.read()
    cfg_w48_384_288 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 14
    model = hrnet.get_pose_net(cfg_w48_384_288, is_train=False)
    model = model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ3\snapshot_PoseHRNet-W48_288x384.pth", weights_only=True, map_location=device))
    table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    for confidence in confidence_range:
        rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
                test_pose(model, image_test_folder, annotation_path,
                          input_size=cfg_w48_384_288['MODEL']['IMAGE_SIZE'],
                          output_size=cfg_w48_384_288['MODEL']['HEATMAP_SIZE'],
                          device=device,
                          output_folder=output_folder,
                          confidence=confidence)
        table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
                                                 "RMSE": rmse, 
                                                 "coverage": coverage, 
                                                 "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
                                                 "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
                                                 ignore_index=True)
    table.to_csv(os.path.join(output_folder, 'hrnet_w48_384_288-RMSEs.csv'), index=False)

    # -------------------------------
    # ----------- ResNet-50
    # -------------------------------

    # with open(r'marco\config\res50_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f: yaml_text = f.read()
    # cfg_res50_256x192 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    # cfg_res50_256x192['MODEL']['NUM_JOINTS'] = 14
    # model = resnet.get_pose_net(cfg_res50_256x192, is_train=False)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ1\snapshot_PoseResNet-50_192x256.pth", weights_only=True, map_location=device))
    # table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    # for confidence in confidence_range:
    #     rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
    #             test_pose(model, image_test_folder, annotation_path,
    #                       input_size=cfg_res50_256x192['MODEL']['IMAGE_SIZE'],
    #                       output_size=cfg_res50_256x192['MODEL']['HEATMAP_SIZE'],
    #                       device=device,
    #                       output_folder=output_folder,
    #                       confidence=confidence)
    #     table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
    #                                              "RMSE": rmse, 
    #                                              "coverage": coverage, 
    #                                              "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
    #                                              "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
    #                                              ignore_index=True)
    # table.to_csv(os.path.join(output_folder, 'res50_256x192-RMSEs.csv'), index=False)

    # with open(r'marco\config\res50_384x288_d256x3_adam_lr1e-3.yaml', 'r') as f: yaml_text = f.read()
    # cfg_res50_384x288 = yaml.load(yaml_text, Loader=yaml.SafeLoader)
    # cfg_res50_384x288['MODEL']['NUM_JOINTS'] = 14
    # model = resnet.get_pose_net(cfg_res50_384x288, is_train=False)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r"C:\Users\marco\Desktop\trained_models\pose_RQ1\snapshot_PoseResNet-50_288x384.pth", weights_only=True, map_location=device))
    # table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    # for confidence in confidence_range:
    #     rmse, rmse_per_keypoint, coverage, coverage_per_keypoint = \
    #             test_pose(model, image_test_folder, annotation_path,
    #                       input_size=cfg_res50_384x288['MODEL']['IMAGE_SIZE'],
    #                       output_size=cfg_res50_384x288['MODEL']['HEATMAP_SIZE'],
    #                       device=device,
    #                       output_folder=output_folder,
    #                       confidence=confidence)
    #     table = pd.concat([table, pd.DataFrame([{"confidence": confidence, 
    #                                              "RMSE": rmse, 
    #                                              "coverage": coverage, 
    #                                              "rmse_per_keypoint" : rmse_per_keypoint.tolist(),
    #                                              "coverage_per_keypoint": coverage_per_keypoint.tolist()}])], 
    #                                              ignore_index=True)
    # table.to_csv(os.path.join(output_folder, 'res50_384x288-RMSEs.csv'), index=False)