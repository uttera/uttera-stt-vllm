#!/bin/bash
# Uttera STT vLLM — Asset Provisioning
# Version: 0.1.0
# Description: Pre-downloads the configured Whisper model (or the default) from
#              HuggingFace into the local assets/models/huggingface cache so the
#              first request does not pay a ~1 GB download at cold start.

set -e

echo "🦾 J.A.R.V.I.S. - Provisioning Uttera STT assets..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ASSETS_DIR="$SCRIPT_DIR/assets"
MODELS_DIR="$ASSETS_DIR/models/huggingface"

mkdir -p "$MODELS_DIR"

# Point HuggingFace at the local cache so the download lands in assets/.
export HF_HOME="$MODELS_DIR"

MODEL_NAME="${WHISPER_MODEL:-openai/whisper-large-v3-turbo}"

echo "[*] Pre-downloading model into $MODELS_DIR ..."
echo "    -> model: $MODEL_NAME"

# Prefer the venv's Python if present.
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON_BIN=python3
fi

"$PYTHON_BIN" - <<EOF
import os, sys
from huggingface_hub import snapshot_download
model = os.environ.get("WHISPER_MODEL", "$MODEL_NAME")
print(f"    -> Downloading/verifying '{model}'...")
path = snapshot_download(repo_id=model)
print(f"    [✓] Cached at {path}")
EOF

echo "✅ Uttera STT asset provisioning complete."
