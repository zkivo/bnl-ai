#!/bin/bash

cd "$1"

pwd=$(pwd)

for file in *.mkv; do
    # Get the file name without extension
    filename="${file%.*}"
    
    # Convert mkv to mp4 using FFmpeg
    ffmpeg -i "$file" -c copy "$filename.mp4"
done

cd "$pwd"
