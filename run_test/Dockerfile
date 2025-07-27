# pull base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility,compute,graphics \
    PATH=$PATH:/opt/conda/bin \
    DEBIAN_FRONTEND=noninteractive \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.4.1

WORKDIR /opt

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libopencv-dev \
        tzdata \
        locales \
        apt-utils &&\
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

# install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh -P ./downloads/ && \
    bash ./downloads/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf ./downloads

# install additional libraries
COPY requirements.txt .
RUN pip install -U pip && \
    pip install pycocotools==2.0.8 && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip \
    rm -f requirements.txt

RUN locale-gen ja_JP.UTF-8

ENV TZ="Asia/Tokyo" \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:en

RUN useradd -m signate
USER signate
WORKDIR /workspace