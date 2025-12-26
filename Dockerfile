# Use the official NVIDIA CUDA base image with Python 3.10 and cu12.4
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, system dependencies, and pip
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-all poppler-utils libreoffice libreoffice-common \
    build-essential python3.10-dev unixodbc-dev \
    python3.10 python3.10-distutils curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libodbc2\
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app
# Make sure CUDA is discoverable
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy requirements and install dependencies
COPY requirements.txt .
# RUN pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# RUN pip install --no-deps --no-build-isolation flash-attn
RUN pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
RUN pip install psutil
RUN pip install --no-cache-dir --no-deps --no-build-isolation -r requirements.txt
# RUN pip install -v -e .

# ENV HF_HOME=/app/custom_models/

# ENV NCCL_P2P_DISABLE=1 \
#     NCCL_IB_DISABLE=1 \
#     NCCL_SHM_DISABLE=1 \
#     NCCL_SOCKET_IFNAME=lo \
#     NCCL_LAUNCH_MODE=GROUP

# # Create a big temp area on /data and set as default tmp
# ENV TMPDIR=/data/tmp \
#     TEMP=/data/tmp \
#     TMP=/data/tmp

# Put common caches on /data too (keeps / small)
# ENV XDG_CACHE_HOME=/data/.cache \
#     PIP_CACHE_DIR=/data/.cache/pip \
#     HF_HOME=/data/hf 

ENV NLTK_DATA=/usr/local/nltk_data

# Pre-download NLTK tokenizers so runtime doesn't try to fetch them
# RUN python - <<'PY'
# import nltk, os
# os.makedirs("/usr/local/nltk_data", exist_ok=True)
# for pkg in ("punkt","stopwords"):
#     nltk.download(pkg, download_dir="/usr/local/nltk_data")
# # try optional resources; ignore if not available
# for pkg in ("punkt_tab","averaged_perceptron_tagger_eng","averaged_perceptron_tagger"):
#     try: nltk.download(pkg, download_dir="/usr/local/nltk_data")
#     except Exception: pass
# PY

ENV HF_HOME=/app/custom_models/

# RUN docling-tools models download
# Copy the rest of the project
COPY . .

# Default command
CMD ["python"]












