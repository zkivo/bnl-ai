# bnl-ai
Tools and AI software for the Behavioural Neuroscience Laboratory.

The video below shows the keypoint predictions of body parts using HRNet. First, we predict the bounding boxes using YOLOv11, and then we predict the body part coordinates using HRNet. Next, we calculate the body angle and plot it on the side.

https://github.com/user-attachments/assets/b983264f-6121-4cea-bcec-9a3c00894cc5


## Fix MKV files
Usually the recordings are corrupted. The timestaps are *not monotonically increasing*. This causes errors in the
processing of the file. To adjust the timestaps use the following command:

```ffmpeg -fflags +genpts -i input.mkv -c:v copy -c:a copy -avoid_negative_ts make_zero input.mp4```

### Fix MKV files given *root folder*
This code will go recursively inside the root folder and will fix all the .mkv files.

Bash:
```Bash:
find /path/to/root -type f -name "*.mkv" -exec sh -c 'ffmpeg -fflags +genpts -i "$1" -c:v copy -c:a copy -avoid_negative_ts make_zero "${1%.mkv}.mp4"' _ {} \;
```

Powershell:
```Powershell
Get-ChildItem -Path "C:\path\to\root" -Recurse -Filter "*.mkv" | ForEach-Object {
    $inputFile = $_.FullName
    $outputFile = [System.IO.Path]::ChangeExtension($inputFile, ".mp4")
    ffmpeg -fflags +genpts -i "$inputFile" -c:v copy -c:a copy -avoid_negative_ts make_zero "$outputFile"
}
```


## Calibration example

```python .\extract_frames.py -i "path/to/calibration/videos" -o "path/to/root/folder/output" -m linear -t 500 ```

## Multicamera Contours + K-Means Extraction 

```python .\extract_frames.py -i "path/to/root/folder/videos" -r -o "path/to/root/folder/output" -m contours -n 50 -g "^(.*)(?=_[0-5]\.mkv)" ```

## Adjust Brightness
```python .\adjust_brightness.py "path/to/root/folder/frames" "path/to/root/folder/output0" 150```
