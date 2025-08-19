
# Download FFmpeg
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$downloadPath = "$env:TEMP\ffmpeg.zip"
$extractPath = "C:\ffmpeg"

Write-Host "Downloading FFmpeg..."
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath

Write-Host "Extracting FFmpeg..."
Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force

# Find the bin directory
$binPath = Get-ChildItem -Path $extractPath -Recurse -Directory -Name "bin" | Select-Object -First 1
$fullBinPath = Join-Path $extractPath $binPath

Write-Host "FFmpeg extracted to: $fullBinPath"
Write-Host "To add to PATH, run as Administrator:"
Write-Host "[Environment]::SetEnvironmentVariable('PATH', $env:PATH + ';$fullBinPath', 'Machine')"
