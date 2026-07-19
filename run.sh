#!/usr/bin/env bash
# Bootstrap and launch the Dubins Path Demonstrator (Linux/macOS).
#
# On first run this creates a local virtualenv in .venv, installs the app into
# it, and starts the GUI. Subsequent runs reuse the venv and launch instantly.
#
# Requires Python >= 3.12 with Tkinter. On Debian/Ubuntu install Tkinter with:
#   sudo apt install python3-tk
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating virtualenv in $VENV ..."
  "$PYTHON" -m venv "$VENV"
fi

# Install (or refresh) the app; quiet unless something actually changes.
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e .

echo "Launching Dubins Path Demonstrator ..."
exec "$VENV/bin/dubins-demo"
