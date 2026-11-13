$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& "$projectRoot\.venv\Scripts\python.exe" "$PSScriptRoot\download_ohlcv.py" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
