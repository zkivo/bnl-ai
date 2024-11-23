# Save this script as convert-mkv-to-mp4.ps1
param (
    [string]$directoryPath
)

# Check if the first argument is provided and if it's an existing directory
if (-not (Test-Path -Path $directoryPath -PathType Container)) {
    Write-Host "The specified path '$directoryPath' does not exist or is not a directory." -ForegroundColor Red
    exit 1
}

# Retrieve all .mkv file paths in the directory and save them in a list
Write-Host "Scanning directory for .mkv files..." -ForegroundColor Green
$mkvFiles = Get-ChildItem -Path $directoryPath -Filter "*.mkv" -File | Select-Object -ExpandProperty FullName

if ($mkvFiles.Count -eq 0) {
    Write-Host "No .mkv files found in the specified directory." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($mkvFiles.Count) .mkv file(s). Starting conversion..." -ForegroundColor Green

# Iterate through the list of .mkv files and convert them to .mp4 using ffmpeg
foreach ($mkvFile in $mkvFiles) {
    $outputPath = [System.IO.Path]::ChangeExtension($mkvFile, ".mp4")
    Write-Host "Converting '$mkvFile' to '$outputPath'..." -ForegroundColor Cyan

    # Run the ffmpeg command
    ffmpeg -err_detect ignore_err -i $mkvFile -map 0 -c copy $outputPath
    # ffmpeg -i $mkvFile -c copy $outputPath

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully converted: $mkvFile -> $outputPath" -ForegroundColor Green
    } else {
        Write-Host "Failed to convert: $mkvFile" -ForegroundColor Red
    }
}

Write-Host "Conversion process completed." -ForegroundColor Green
