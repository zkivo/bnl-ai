# bnl-ai
Tools and AI software for the Behavioural Neuroscience Laboratory.

## Calibration example

python .\extract_frames.py -i "path/to/calibration/videos" -o "path/to/root/folder/output" -m linear -t 500 

## Multicamera Contours + K-Means Extraction 

python .\extract_frames.py -i "path/to/root/folder/videos" -r -o "path/to/root/folder/output" -m contours -n 50 -g "^(.*)(?=_[0-5]\.mkv)" 