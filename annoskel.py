import argparse
import os
import sys
from pathlib import Path
from termcolor import colored

def get_image_files(input_path, recursive):
    """
    Retrieve image files from the input path.

    Args:
        input_path (str): Either a directory or a list of image files.
        recursive (bool): Whether to search recursively in directories.

    Returns:
        list: A list of image file paths.

    """
    if os.path.isdir(input_path):
        pattern = "**/*" if recursive else "*"
        image_files = [str(p) for p in Path(input_path).glob(pattern) if p.is_file()]
        for image in image_files:
            if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(colored(f"The file '{image}' does not end as one of the following format .png, .jpg, .jpeg, .bmp, .tiff", "yellow"))
                image_files.remove(image)
        if not image_files:
            print(colored(f"No image files found in directory: {input_path} (recursive={recursive})", "red"))
            sys.exit(1)
        return image_files
    elif os.path.isfile(input_path):
        if not input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(colored(f"The file '{input_path}' does not end as one of the following format .png, .jpg, .jpeg, .bmp, .tiff", "red"))
            sys.exit(1)
        return [input_path]
    else:
        print(colored(f"Invalid input: {input_path}. Must be a valid file or directory.", "red"))
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AnnoSkel - A terminal software for annotating skeletons.")
    
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="A folder, image, or a list of images for annotation.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Enable recursive processing of folders.",
    )
    parser.add_argument(
        "-c", "--check",
        action="store_true",
        help="Enable checking mode for annotated skeletons.",
    )

    args = parser.parse_args()

    input_images = []
    for item in args.input:
        input_images.extend(get_image_files(item, args.recursive))

    print("=== AnnoSkel ===")
    print(f"Input: {input_images}")
    print(f"Recursive: {'Enabled' if args.recursive else 'Disabled'}")
    print(f"Check Mode: {'Enabled' if args.check else 'Disabled'}")

    # Main functionality placeholder
    print("Processing input...")

if __name__ == "__main__":
    main()
