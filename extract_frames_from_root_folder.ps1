# Script to gather all files from subfolders and pass them to 'extract_frames'

# Check if a root folder is provided as an argument
if (-not $args[0]) {
    Write-Host "Please provide a root folder as an argument."
    exit
}

# Get the root folder from the argument
$RootFolder = $args[0]

# Check if the provided path is a valid directory
if (-not (Test-Path -Path $RootFolder -PathType Container)) {
    Write-Host "The provided path is not a valid directory. Please provide a valid folder path."
    exit
}

# Initialize an empty list to store full paths of files
$FileList = @()

# Recursively gather all files from the subfolders
$FileList = Get-ChildItem -Path $RootFolder -Recurse -File | ForEach-Object { $_.FullName }

# Check if there are any files
if ($FileList.Count -eq 0) {
    Write-Host "No files found in the given directory and its subdirectories."
    exit
}

# Convert the file list to a single string separated by spaces
$FileListString = $FileList -join " "

# Write-Host $FileListString

# Execute the 'extract_frames' program with the file list as an argument
Write-Host "Executing 'extract_frames' with the file list..."
$Command = "python extract_frames.py $FileListString `"C:\Users\marco\Desktop\Extracted Frames`""
Invoke-Expression $Command