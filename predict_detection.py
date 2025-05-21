from ultralytics import YOLO
import os

def list_filenames(folder_path):
    try:
        file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        return file_paths
    except FileNotFoundError:
        print("Error: Folder not found.")
        return []
    except PermissionError:
        print("Error: Permission denied.")
        return []


if __name__ == '__main__':
    # Load a model
    model = YOLO(r"trained_models\detection_Top_1k_300epochs\weights\best.pt")  # pretrained YOLO11n model

    # filenames = list_filenames(r'datasets\top_detection_v1\images\val')
    video_filename = r"C:\Users\marco\Desktop\20250315-151340-413969_2.mkv"

    model.predict(video_filename, show=True, save=True)
    exit()

    # Run batched inference on a list of images
    results = model(video_filename, stream=True)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        print(boxes)
        result.show()  # display to screen
        input()
        # result.save(filename="result.jpg")  # save to disk