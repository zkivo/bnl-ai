import sys
import os
import re
import cv2
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.io import imread
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
from collections import defaultdict
from extraction_methods import contours_extraction

for video_file in video_list:
    print(f"Processing video: {video_file}")
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_file}'.")
        continue

    print("Extracting features with color histogram...")
    frame_count = 0
    frame_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            print(f"frame: {frame_count}", end="\r", flush=True)  # Overwrites the same line and forces flush
        frame_features.append(extract_color_histogram(frame))
        frame_count += 1
    cap.release()

    frame_features = np.array(frame_features)   

    print("Clustering with K-means...")
    # Define the number of clusters (e.g., 25)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(frame_features)

    # Select the frame closest to the centroid of each cluster
    selected_frames = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_features = frame_features[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_frame_index = cluster_indices[np.argmin(distances)]
        selected_frames.append(closest_frame_index)

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_file}'.")
        continue

    basename = os.path.basename(video_file)
    root_name, ext = os.path.splitext(basename)

    frames_folder = os.path.join(out_folder, root_name)
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    print(f"Writing frames to folder {frames_folder}...")
    frame_number = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % 30 == 0:
            print(f"frame: {frame_number}", end="\r", flush=True)  # Overwrites the same line and forces flush
        if frame_number in selected_frames:
            frame_path = os.path.join(frames_folder, f"{root_name}-{frame_number}.png")
            cv2.imwrite(frame_path, frame)
        frame_number += 1
    cap.release()


def extract_color_histogram(frame):
    """
    This function calculates the color histogram for a given frame.
    The histogram for each color has 8 bins, that means that
    the output will have 8+8+8=24 features.
    This is usuful when the difference between frames is mostly
    seen as color differences.
    """
    frame = cv2.resize(frame, (128, 128))  # Resize for consistency
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extract_hog_features(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (128, 128))
    features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features