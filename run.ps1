# Bootstrap and launch the Dubins Path Demonstrator (Windows, PowerShell).
#
# On first run this creates a local virtualenv in .venv, installs the app into
# it, and starts the GUI. Subsequent runs reuse the venv and launch instantly.
#
# Requires Python >= 3.12 with Tkinter (bundled with the standard python.org
# CPython installer for Windows).
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venv = ".venv"

if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Write-Host "Creating virtualenv in $venv ..."
    python -m venv $venv
}

# Install (or refresh) the app; quiet unless something actually changes.
& "$venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& "$venv\Scripts\python.exe" -m pip install --quiet -e .

Write-Host "Launching Dubins Path Demonstrator ..."
& "$venv\Scripts\dubins-demo.exe"
