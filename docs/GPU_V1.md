# GPU experimental branch (CUDA / TensorRT)

This branch is an **experimental first implementation**, not yet a validated high-performance GPU release. Original `main` remains unchanged.

## Backends

- `--backend cuda`: ONNX Runtime CUDA EP; unsupported nodes can fall back to CPU.
- `--backend tensorrt`: TensorRT EP (FP16 enabled) -> CUDA EP -> CPU. TRT engine cache stored in `./trt_engine_cache` relative to the launch working directory; cache is hardware, TensorRT version and graph dependent. On a new GPU or TensorRT build, clear and regenerate cache.
- `--gpu-device 0`: device index.
- `--workers 2`: parallel HTTP request workers; tune 1, 2, 4, 8 through benchmark. Memory and GPU contention may make additional workers slower.
- `--max-work-mb 1024`: approximate per-request working-buffer budget, **not** a global RSS/VRAM limit.

## Build (Windows)

1. Install an NVIDIA GPU driver; install versions of CUDA, cuDNN, and (for TRT) TensorRT compatible with the chosen ONNX Runtime GPU distribution.
2. Obtain the **C++ Windows GPU** ONNX Runtime package of a matching version. The existing locked CPU package is **not** sufficient.
3. Configure with `-DONNXRUNTIME_ROOT=C:\path\to\onnxruntime-gpu` and the existing OpenCV_DIR. Build Release x64.
4. Ensure `onnxruntime_providers_cuda.dll`, optionally `onnxruntime_providers_tensorrt.dll`, and the matching NVIDIA runtime DLLs are discoverable by the loader. The CMake install step copies DLLs found in the selected ORT package lib directory.
5. Copy your original model files into `models/` and retain the matching `config/*.json`.
6. Run `onnx-seg-server.exe --backend cuda --gpu-device 0 --workers 2` or switch to `--backend tensorrt` if TensorRT is properly installed.

## HTTP API

Unchanged from main: `POST /infer` with multipart fields `file`, **`catagory`** (legacy spelling), `item`. Returns PNG bytes and the original timing headers. `GET /health` remains available.

## Important limitations

- **The entire end-to-end pipeline is not on GPU yet.** Image decode, BGR/RGB conversion and normalization, tiles, score restoration, blending, argmax/coloring and PNG encoding still run on CPU. `Ort::Session::Run` uses CPU input and CPU output tensors; host/device transfers are implicit. No I/O binding, GPU kernels, batched tiles, pinned host memory, CUDA Graph or async host/device overlap are implemented yet.
- FP16 is enabled for TensorRT only. CUDA EP may execute FP32 model operators. A model exported FP16 must match the current float32-input/float32-output contract or use compatible wrapper conversions. Validate segmentation accuracy against baseline before claiming equivalent quality.
- Session sharing is retained per model. Multiple worker requests can execute the same session concurrently; actual speed and safety for individual providers/custom operators must be tested.
- The reported `X-Inference-Ms` measures the wall time of `Run` per tile and may include transfers and synchronization; it is **not kernel-only latency**.
- CPU fallback is allowed by ONNX Runtime provider assignment; inspect ORT verbose logs/profiling to ensure nodes actually use the GPU, and monitor with `nvidia-smi`. No claim of full GPU utilization is made.
- There is **no actual GPU benchmark or build verification** in this workspace. Benchmark with your ONNX models on the target GPU.

## Suggested next engineering milestones

1. GPU-native resize + RGB normalization kernel (CUDA), pinned staging buffers and I/O binding.
2. CUDA output buffers and GPU score resize, overlap averaging, argmax and compact mask transfer.
3. Per-model ring of GPU buffers/streams and controlled microbatching of tiles; only if model batch dimension supports it.
4. Optional CUDA Graph capture for static shapes after correctness validation; TRT optimization profile and cache management.
5. Instrument p50/p95/p99 latency, throughput, memory peak and per-stage GPU events, compare CPU, CUDA-FP32, TensorRT-FP16 and calibrated INT8.

## Benchmark

```powershell
python scripts/benchmark_http.py --url http://127.0.0.1:9600/infer --image C:\data\image.png --category 1001 --item 4 --requests 100 --concurrency 2
```

Repeat for 1/2/4/8 workers and for each backend. Warm up first, compare PNG output pixelwise and track total server-side time plus network round trip. Do not compare only single-model `Run()` time.
