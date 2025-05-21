import pandas as pd
import shutil
import os

def convert_bbox(top_left_x, top_left_y, bottom_right_x, bottom_right_y, img_width=1920, img_height=1200):
    center_x = (top_left_x + bottom_right_x) / 2.0 / img_width
    center_y = (top_left_y + bottom_right_y) / 2.0 / img_height
    width = (bottom_right_x - top_left_x) / img_width
    height = (bottom_right_y - top_left_y) / img_height
    return center_x, center_y, width, height

images_path = r'marco\datasets\Top1k\train'
annotation_path  = r'marco\datasets\Top1k\annotations.csv'

images = os.listdir(images_path)
df = pd.read_csv(annotation_path)

# Ensure the required columns exist
required_columns = ['filename', 'bbox_tl-x', 'bbox_tl-y', 'bbox_br-x', 'bbox_br-y']
if not all(col in df.columns for col in required_columns):
    raise ValueError("CSV file is missing one or more required columns.")

# Define output folder
out_labels = r'marco\datasets\Top_detection_1k\labels\train'
out_images = r'marco\datasets\top_detection_1k\images\train'
os.makedirs(out_labels, exist_ok=True)
os.makedirs(out_images, exist_ok=True)

# Process each row and save bounding box information
for _, row in df.iterrows():
    image_filename = row['filename']
    if image_filename not in images:
        continue
    shutil.copy(os.path.join(images_path, image_filename), out_images)
    
    center_x, center_y, width, height = convert_bbox(
        row['bbox_tl-x'], row['bbox_tl-y'], row['bbox_br-x'], row['bbox_br-y']
    )
    output_filename = os.path.join(out_labels, f"{os.path.splitext(image_filename)[0]}.txt")
    with open(output_filename, 'a') as f:
        f.write(f"0 {center_x} {center_y} {width} {height}\n")

print(f"Bounding box annotations saved in: {out_labels}")
