import deeplabcut as dlc
from napari import run
import sys
import os

# for some reason, label_frames does not work. 
# The Napari window closes immediately after opening.

def main():
    if len(sys.argv) != 2:
        print("Usage: python label_frames.py <project_directory>")
        sys.exit(1)

    project_directory = sys.argv[1]
    
    dlc.label_frames(os.path.join(project_directory, 'config.yaml'))
    run()

if __name__ == "__main__":
    main()