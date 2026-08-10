# Immich Machine Learning (v3.1.0) — Jetson build
#
# CUDA-accelerated ML server for NVIDIA Jetson (tested: Orin Nano Super,
# JetPack 7 / L4T R39.2, driver 595.78). Based on upstream
# immich-app/immich machine-learning/Dockerfile (AGPL-3.0).
#
# Build on an arm64/Jetson host:
#   docker build -t immich-ml-jetson:3.1.0 .
#
# Notes:
#  - ONNX Runtime GPU wheel is the NVIDIA Jetson build (Jetson AI Lab devpi,
#    jp6/cu129); it requires glibc >= 2.38, hence the trixie prod base.
#  - The image intentionally does NOT preload mimalloc and does NOT use CUDA
#    toolkit "compat" lib paths (both break CUDA on Jetson).

FROM python:3.12-bookworm@sha256:3cd9086bdb30f7c9bc08a3fa621d9842e0d3f6f9291aeb4677e0547817c10b12 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0 /uv /uvx /bin/

WORKDIR /src
COPY uv.lock pyproject.toml ./
RUN sed -i \
      's|https://pypi.org/simple|https://mirrors.aliyun.com/pypi/simple|g; s|https://files.pythonhosted.org/packages|https://mirrors.aliyun.com/pypi/packages|g' \
      uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-project \
      --compile-bytecode --no-progress --active --link-mode copy

# NVIDIA Jetson ONNX Runtime GPU wheel (sha256-pinned; override the URL for
# mirrors/offline builds).
ARG ORT_WHEEL_URL=https://pypi.jetson-ai-lab.io/jp6/cu129/+f/2e3/a07114007df15/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl
ARG ORT_WHEEL_SHA256=2e3a07114007df15db673852d798d6f47f91362f0ac084d6fa04e414a06dc25e
RUN curl -fL --retry 3 -o /tmp/ort.whl "${ORT_WHEEL_URL}" && \
    echo "${ORT_WHEEL_SHA256}  /tmp/ort.whl" | sha256sum -c - && \
    mkdir -p /wheels && mv /tmp/ort.whl /wheels/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl
RUN uv pip install --no-deps /wheels/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl

# CUDA 12.x runtime libs for the ORT CUDA EP (cublas/cudnn/cudart/curand/cufft)
RUN uv pip install --index-url https://mirrors.aliyun.com/pypi/simple/ \
      nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 \
      nvidia-curand-cu12 nvidia-cufft-cu12

FROM python:3.12-slim-trixie@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS prod

ENV MACHINE_LEARNING_MODEL_ARENA=false \
    TRANSFORMERS_CACHE=/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/usr/src \
    VIRTUAL_ENV=/opt/venv \
    MACHINE_LEARNING_CACHE_FOLDER=/cache \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu"

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends tini libgl1 libglib2.0-0 libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src
COPY --from=builder /opt/venv /opt/venv
COPY scripts/healthcheck.py .
COPY immich_ml immich_ml

ENV IMMICH_REPOSITORY=immich-app/immich \
    IMMICH_REPOSITORY_URL=https://github.com/immich-app/immich

LABEL org.opencontainers.image.source="https://github.com/zhzy0077/immich-ml-jetson" \
      org.opencontainers.image.licenses="AGPL-3.0"

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "immich_ml"]

HEALTHCHECK CMD python3 healthcheck.py
