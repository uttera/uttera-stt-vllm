# Uttera STT vLLM — container image
# SPDX-License-Identifier: Apache-2.0
#
# Build:
#   docker build -t uttera-stt-vllm:0.1.0 .
# Run (with GPU):
#   docker run --gpus all --rm -p 9005:9005 \
#       -e WHISPER_MODEL=openai/whisper-large-v3-turbo \
#       uttera-stt-vllm:0.1.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        git \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install Python deps first (layer cache-friendly).
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the rest of the project.
COPY . /app/

EXPOSE 9005

# Model cache defaults to assets/models/huggingface inside the image; mount a
# volume there to persist downloads across container rebuilds.
ENV XDG_CACHE_HOME=/app/assets/models

CMD ["uvicorn", "main_stt:app", "--host", "0.0.0.0", "--port", "9005"]
