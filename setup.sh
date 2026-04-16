#!/bin/bash
# Uttera STT vLLM — Unified Setup Script
# Version: 0.1.0
# Description: Orchestrates Python environment setup. Python 3.12 is the
#              reference target (vLLM 0.19 supports 3.9-3.12).

set -e

echo "🦾 J.A.R.V.I.S. - Starting Uttera STT vLLM installation..."

# 1. Python Virtual Environment
echo "[*] Initialising Python Virtual Environment..."
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
    echo "    -> Using python3.12"
else
    PYTHON_BIN=python3
    echo "    [!] python3.12 not found, falling back to $(python3 --version)."
fi
$PYTHON_BIN -m venv venv
source venv/bin/activate

# 2. Build-time dependencies
echo "[*] Installing build-time dependencies..."
pip install --upgrade pip setuptools wheel

# 3. Core Dependencies
echo "[*] Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Trigger Asset Provisioning (pre-download the default model)
if [ -f "./setup_assets.sh" ]; then
    echo "[*] Python environment ready. Handing over to setup_assets.sh..."
    chmod +x setup_assets.sh
    ./setup_assets.sh
else
    echo "[!] WARNING: setup_assets.sh not found. Models will be downloaded on first request instead."
fi

echo "✅ All systems operational."
