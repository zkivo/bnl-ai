import cv2
import numpy as np
import os
import sys

def adjust_brightness(image, target_brightness=150):
    """
    Adjusts the brightness of an image to a target brightness level.

    :param image: Input image as a NumPy array.
    :param target_brightness: Desired average brightness level (0-255).
    :return: Brightened image as a NumPy array.
    """
    # Convert to grayscale to calculate current brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)

    # Calculate the brightness factor
    brightness_factor = target_brightness / current_brightness

    # Adjust the brightness
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    return brightened_image

def process_images_from_folders(input_root_folders, output_root_folder, target_brightness=150):
    """
    Adjusts the brightness of all images in multiple folders and saves the results.

    Arguments
    ---------
        input_folders: 
            List of paths to folders containing input images.

        output_root_folder: 
            Path to the root folder to save brightened images.

        target_brightness: 
            Desired average brightness level (0-255).
    """
    for folder in input_folders:
        folder_name = os.path.basename(folder.rstrip('/\\'))  # Get folder name
        output_folder = os.path.join(output_root_folder, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Read the image
                img_path = os.path.join(folder, filename)
                image = cv2.imread(img_path)

                # Adjust brightness
                brightened_image = adjust_brightness(image, target_brightness)

                # Save the image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, brightened_image)

if __name__ == "__main__":
    process_images_from_folders(sys.argv[1] , sys.argv[2], sys.argv[3])
    print("Brightness adjustment complete.")