# Style-Bert-VITS2 TTS Inference Server (GPU)
# Adapted from Dockerfile.deploy for NVIDIA CUDA GPU inference.
# Uses server_fastapi.py (API server) — not the editor or trainer.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Minimal system deps: Python 3.10 + ffmpeg + git
# No gcc/cmake needed — pyopenjtalk-dict ships prebuilt manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid resolver assertion bugs in the stock Ubuntu 22.04 version
RUN pip install --upgrade pip

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# --- Install Python dependencies (layer-cached separately from source) ---

# 1) PyTorch with CUDA 11.8 (pinned <2.4 — Style-Bert-VITS2 models are incompatible with newer torch)
RUN pip install --no-cache-dir "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118

# 2) TTS inference-only requirements
COPY requirements-tts.txt /app/requirements-tts.txt
RUN pip install --no-cache-dir -r requirements-tts.txt

# --- Copy source code ---
COPY . /app/

# --- Download models (BERT models + default JP voice) ---
# --only_infer skips SLM/pretrained training models (not needed for inference).
# BERT models go to bert/, voice models go to model_assets/.
# This downloads ~6.8 GB during build instead of copying ~8 GB from local context.
RUN python initialize.py --only_infer

# --- Pre-fetch the OpenJTalk system dictionary at BUILD time ---
# pyopenjtalk lazily downloads + extracts open_jtalk_dic_utf_8-1.11 on first use.
# A runtime download is fragile: if extraction is interrupted the dict dir ends up
# partial (missing char.bin / unk.dic) and pyopenjtalk never re-extracts, which
# causes a permanent "Failed to initialize Mecab" crash loop on server startup.
# Placed AFTER initialize.py so all heavy layers stay cached and the rebuild is fast.
# Bake the complete dict into the image and fail the build if critical files are missing.
RUN python -c "import pyopenjtalk, os; pyopenjtalk.unset_user_dict(); d=os.path.join(os.path.dirname(pyopenjtalk.__file__),'open_jtalk_dic_utf_8-1.11'); assert os.path.isfile(os.path.join(d,'char.bin')), 'char.bin missing after extract'; assert os.path.isfile(os.path.join(d,'unk.dic')), 'unk.dic missing after extract'; print('OpenJTalk dict baked & verified:', sorted(os.listdir(d)))"

# Default synthesis parameters
ENV DEFAULT_MODEL=jvnv-F1-jp
ENV DEFAULT_STYLE=Neutral
ENV DEFAULT_SDP_RATIO=0.2
ENV DEFAULT_LENGTH=1.0

# Expose API server port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/models')" || exit 1

# Start the API server
CMD ["python", "server_fastapi.py"]
