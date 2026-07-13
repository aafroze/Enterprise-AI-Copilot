#!/usr/bin/env pwsh
# Load environment variables from .env and launch Streamlit

# Read .env file and set environment variables
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -notmatch "^#" -and $_ -notmatch "^$" } | ForEach-Object {
        $key, $value = $_ -split "=", 2
        if ($key -and $value) {
            $env:$key = $value
            Write-Host "✓ Set $key"
        }
    }
    Write-Host ""
}

# Launch Streamlit
python -m streamlit run app.py
