$output_dir = "D:\mouse frames\top"
$inputFile = "D:\mouse recordings\20250315- M413966 - Bl6 - M - 20240802\video-14-59-34_2.mkv"
$outputFile = [System.IO.Path]::ChangeExtension($inputFile, ".mp4")

ffmpeg -fflags +genpts -i "$inputFile" -c:v copy -c:a copy -avoid_negative_ts make_zero "$outputFile"

python .\tools\extract_frames.py -i "$outputFile" -o "$output_dir" -m linear -n 30
python .\tools\extract_frames.py -i "$outputFile" -o "$output_dir" -m uniform -n 30