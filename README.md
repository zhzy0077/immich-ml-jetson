# immich-ml-jetson

CUDA-accelerated **Immich Machine Learning** container for **NVIDIA Jetson**,
built from the official [immich-app/immich](https://github.com/immich-app/immich)
release source (v3.2.2). Available as
`ghcr.io/zhzy0077/immich-ml-jetson:3.2.2`.

This is a **Jetson/L4T build** — it is *not* the generic amd64 or ARM-server
CUDA image (Jetson needs the L4T driver stack, `--runtime=nvidia`, and the
NVIDIA Jetson ONNX Runtime wheel).

## Tested environment

- NVIDIA Jetson **Orin Nano Super** Dev Kit, 8 GB
- JetPack 7.x / L4T **R39.2**, driver 595.78, CUDA 13.2 userland
- Docker 29 + nvidia-container-runtime 1.19.1

Verified on JetPack 6.x-style CUDA 12.x stacks should also work (host driver
must be >= the container's CUDA 12.9 userland).

## Image contents

- Immich machine-learning **v3.2.2** (`python -m immich_ml`, port 3003)
- Python 3.12, `uv`-synced dependencies
- **ONNX Runtime GPU 1.23.0** (NVIDIA Jetson AI Lab wheel, `jp6/cu129`,
  sha256-pinned in the Dockerfile) → CUDA + TensorRT + CPU execution providers
  (newest `jp6/cu129` cp312 wheel; upstream asks for `>=1.23.2` on servers,
  Jetson wheel installed with `--no-deps`)
- Upstream v3.2.x no longer depends on `insightface` (vendored SCRFD decode +
  ArcFace align + direct OCR pipeline); new runtime dep is `onnx>=1.22.0`
- `HF_HOME=/cache/hf-cache` (matches upstream, rootless single-volume deploys)
- CUDA 12.9 userland (cublas / cuDNN 9 / cudart) from PyPI wheels
- No mimalloc preload, no CUDA-toolkit `compat/` lib paths (both break CUDA on
  Jetson)

## Build

```bash
# on an arm64/Jetson host
docker build -t ghcr.io/zhzy0077/immich-ml-jetson:3.2.2 .
```

`docker build` needs network access to Docker Hub (bases), ghcr.io (uv binary),
the aliyun PyPI mirror and the NVIDIA Jetson AI Lab devpi (ORT wheel). For
offline/China builds, pre-download the ORT wheel and pass
`--build-arg ORT_WHEEL_URL=file:///path/to/onnxruntime_gpu-...whl` (not supported
for `file://` in RUN; use a local HTTP URL or a build with the wheel copied into
the context instead).

## Run (Jetson)

Jetson requires `--runtime=nvidia`, the host L4T driver userspace, and the GPU
device set that the L4T container runtime would auto-mount for L4T images:

```bash
docker run -d --name immich-ml \
  --runtime=nvidia --restart unless-stopped -p 3003:3003 \
  -e MACHINE_LEARNING_WORKERS=2 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v immich-ml-cache:/cache \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro \
  -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro \
  -v /opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1.1:ro \
  --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl --device /dev/nvidia-modeset \
  --device /dev/nvmap --device /dev/nvsciipc --device /dev/dri --device /dev/nvgpu \
  --device /dev/nvhost-as-gpu --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-ctxsw-gpu \
  --device /dev/nvhost-dbg-gpu --device /dev/nvhost-gpu --device /dev/nvhost-nvsched-gpu \
  --device /dev/nvhost-power-gpu --device /dev/nvhost-prof-ctx-gpu --device /dev/nvhost-prof-dev-gpu \
  --device /dev/nvhost-prof-gpu --device /dev/nvhost-sched-gpu --device /dev/nvhost-tsg-gpu \
  ghcr.io/zhzy0077/immich-ml-jetson:3.2.2
```

Then point your Immich server at `http://<jetson-ip>:3003` as the remote ML
server.

## Verify CUDA is active

```bash
curl http://localhost:3003/ping          # pong
docker logs immich-ml | grep -A2 "Setting execution providers"
# → ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Measured on the Orin Nano Super: CLIP ViT-B/32 visual **~27 ms/image** (CUDA),
~0.39 s/photo end-to-end via API (12 MP JPEG decode + embed); buffalo_l face
detection ~0.49 s/photo.

## Notes / limitations

- Models are downloaded at first use into `/cache` (hf-mirror endpoint used
  above; huggingface.co is blocked in some networks). Use a named volume.
- The container is **stateless**: all job state lives in the Immich server +
  Postgres; only the model cache persists (recreatable).
- The NVIDIA Jetson ONNX Runtime wheel requires **glibc >= 2.38**, hence the
  Debian trixie runtime base.
- If your host's driver layout differs (e.g., older JetPack with
  `/usr/lib/aarch64-linux-gnu/tegra`), adjust the libcuda/nvidia mounts to your
  host's actual paths.

## License

Derived from [immich-app/immich](https://github.com/immich-app/immich)
(AGPL-3.0); the ONNX Runtime GPU wheel is NVIDIA's Jetson build.
