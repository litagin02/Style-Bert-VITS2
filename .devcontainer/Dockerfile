# Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set ENV variables
ENV LANG C.UTF-8
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade"
ENV GIT_CLONE="git clone --depth 10"

# Install basic utilities and dependencies
RUN apt-get update && \
    $APT_INSTALL \
    sudo \
    build-essential \
    ca-certificates \
    wget \
    curl \
    git \
    git-lfs \
    zip \
    unzip \
    nano \
    ffmpeg \
    software-properties-common \
    gnupg \
    python3 \
    python3-pip \
    python3-dev

# Add symlink for Python
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Installing CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    $APT_INSTALL cuda && \  
    rm cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb

# Installing CUDNN
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt-get update && \
    $APT_INSTALL libcudnn8=8.9.7.29-1+cuda12.2 \
    libcudnn8-dev=8.9.7.29-1+cuda12.2

# Set CUDA environment variables
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install PyTorch with CUDA support
RUN $PIP_INSTALL torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Jupyter for development
RUN $PIP_INSTALL jupyterlab

# Copy requirements.txt and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN $PIP_INSTALL -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Setup working directory
WORKDIR /workspace

# Expose ports for Jupyter and Web UI
EXPOSE 7860 8888

# Default command
CMD ["bash"]
