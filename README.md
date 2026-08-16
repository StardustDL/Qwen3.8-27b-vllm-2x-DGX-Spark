# Qwen3.8 27B – vLLM on 2× DGX Spark

Deploy [Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — a multimodal reasoning model — across **two NVIDIA DGX Spark** nodes with vLLM, InfiniBand interconnect, and FP8 KV-cache.

> This project is a modification of [DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context), adapted to serve Qwen3.8 27B with multimodal (vision) support.

## Overview

This repository provides a ready-to-run `docker-compose.yml` and helper scripts that launch a vLLM inference server serving Qwen3.8 27B. The setup uses:

- **2 × DGX Spark** nodes (spark1 = head, spark2 = worker)
- **Tensor parallelism (TP)** across both GPUs
- **InfiniBand** (NCCL over IB) for inter-node communication
- **FP8 KV-cache** (`--kv-cache-dtype fp8`) for memory efficiency
- **Multi-Token Prediction (MTP)** speculative decoding (4 draft tokens)
- **Prefix caching** and **FlashInfer autotune**
- **Multimodal** support (vision encoder, `--mm-encoder-tp-mode data`)
- **Tool calling** and **reasoning** support (Qwen3 parsers)

## Benchmark

Measured with [llama-benchy](https://github.com/eugr/llama-benchy), using `--latency-mode generation` (concurrency = 1, `pp` = prompt processing, `tg` = text generation, `depth` = context depth):

| model | test | t/s | peak t/s | TTFT (ms) | est. PPT (ms) | e2e TTFT (ms) |
|:------|:-----|----:|---------:|----------:|--------------:|--------------:|
| vlm | pp4096 | 2635.62 ± 166.47 | | 1573.26 ± 81.55 | 1435.23 ± 81.55 | 1573.55 ± 81.91 |
| vlm | tg256 | 32.83 ± 2.05 | 42.00 ± 3.56 | | | |
| vlm | pp4096 @ d4096 | 2185.74 ± 30.09 | | 3589.97 ± 72.43 | 3451.94 ± 72.43 | 3589.97 ± 72.43 |
| vlm | tg256 @ d4096 | 28.54 ± 0.94 | 45.33 ± 4.78 | | | |
| vlm | pp4096 @ d8192 | 2128.34 ± 37.51 | | 5419.60 ± 101.28 | 5281.57 ± 101.28 | 5419.60 ± 101.28 |
| vlm | tg256 @ d8192 | 31.26 ± 1.23 | 44.00 ± 2.45 | | | |

## Requirements

### Hardware

| Component | Requirement |
|-----------|------------|
| Nodes     | 2 × DGX Spark (Grace Hopper, GB10) |
| Interconnect | InfiniBand (e.g., ConnectX-7) between nodes |
| Storage   | Sufficient for model weights (~30 GB with HF cache) |

### Software

- **Docker** with `docker compose` plugin (v2.24+)
- **Passwordless SSH** from spark1 → spark2
- **NVIDIA Container Toolkit** (`nvidia-ctk`) — installed by default on DGX Spark
- **InfiniBand drivers** (`ibdev2netdev`, `ibstat`) — installed by default
- **Git** and **curl**

## Quick Start

### 1. Clone the repo on both nodes

```bash
git clone https://github.com/StardustDL/Qwen3.8-27b-vllm-2x-DGX-Spark.git Qwen3.8-27b-vllm-2x-DGX-Spark
cd Qwen3.8-27b-vllm-2x-DGX-Spark
```

Run this on **both** spark1 (head) and spark2 (worker) and clone them into the same path.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to match your cluster:

| Variable | Description | Example |
|----------|-------------|---------|
| `NODE_RANK` | `0` for head (spark1), `1` for worker (spark2) | `0` |
| `HEADLESS` | Set to `1` on worker nodes | *(empty for head)* |
| `MASTER_ADDR` | CX7 IP address of the head node (spark1) | `10.100.80.1` |
| `WORKER_HOST` | CX7 IP address of the worker node (spark2) | `10.100.80.2` |
| `HF_CACHE` | Path to your HuggingFace cache | `${HOME}/.cache/huggingface` |
| `NCCL_IB_HCA` | InfiniBand HCA device (run `ibdev2netdev -v`) | `rocep1s0f1` |
| `NCCL_SOCKET_IFNAME` | Network interface for socket comms | `enp1s0f1np1` |
| `GPU_MEMORY_UTILIZATION` | GPU memory budget (fraction) | `0.4` |
| `MAX_MODEL_LEN` | Maximum context length (tokens) | `262144` |
| `MASTER_PORT` | NCCL master port | `25000` |
| `VLLM_PORT` | vLLM API port | `8000` |

### 4. Start the server

From **spark1** only:

```bash
./start.sh
```

This script:
1. Copies `.env`, `docker-compose.yml`, and the helper scripts to spark2
2. SSHs into spark2 and starts the container there (`NODE_RANK=1`, `HEADLESS=1`)
3. Starts the container on spark1
4. Polls `http://127.0.0.1:8000/v1/models` until the API is ready (up to ~20 minutes)

### 5. Stop the server

From **spark1** only:

```bash
./stop.sh
```

## API Usage

Once running, the OpenAI-compatible vLLM API is available on both nodes at `http://localhost:8000` (or your configured `VLLM_PORT`).

### Chat completion (with reasoning)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlm",
    "messages": [
      {"role": "user", "content": "What is 47 × 89?"}
    ],
    "temperature": 0.0
  }'
```

### Streaming

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlm",
    "messages": [
      {"role": "user", "content": "Write a haiku about distributed inference"}
    ],
    "stream": true
  }'
```

### Multimodal (image input)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vlm",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
          {"type": "text", "text": "Describe this image."}
        ]
      }
    ]
  }'
```

### Tool calling

The server is configured with `--tool-call-parser qwen3_coder` and `--enable-auto-tool-choice` for function-calling tasks.

## Architecture

```
┌─────────────────────────┐       InfiniBand        ┌─────────────────────────┐
│     spark1 (head)       │◄───────────────────────►│    spark2 (worker)      │
│  NODE_RANK=0            │   NCCL (IB / sockets)    │  NODE_RANK=1            │
│  ┌───────────────────┐  │                          │  ┌───────────────────┐  │
│  │ vLLM container    │  │                          │  │ vLLM container    │  │
│  │ TP rank 0 (GPU 0) │  │                          │  │ TP rank 1 (GPU 1) │  │
│  │ Port 8000 (API)   │  │                          │  │ Port 8000 (API)   │  │
│  └───────────────────┘  │                          │  └───────────────────┘  │
└─────────────────────────┘                          └─────────────────────────┘
```

### Key vLLM parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tensor-parallel-size` | `2` | Split model across 2 GPUs |
| `--pipeline-parallel-size` | `1` | No pipeline parallelism |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache for memory savings |
| `--max-model-len` | `262144` (default) | Max context length |
| `--max-num-seqs` | `4` | Max concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Max tokens per batch |
| `--block-size` | `256` | PagedAttention block size |
| `--gpu-memory-utilization` | `0.2` (default) | GPU memory budget |
| `--enable-prefix-caching` | enabled | Reuse KV cache across requests |
| `--speculative-config` | MTP, 4 tokens | Multi-Token Prediction |
| `--distributed-executor-backend` | `mp` | Multi-process backend |
| `--nnodes` | `2` | Two-node deployment |
| `--mm-encoder-tp-mode` | `data` | Multimodal encoder tensor parallelism |
| `--tool-call-parser` | `qwen3_coder` | Qwen3 tool calling |
| `--reasoning-parser` | `qwen3` | Qwen3 reasoning format |

## Docker Image

The compose file uses `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` — a nightly vLLM build optimized for DGX Spark (GB10, CUDA arch 12.1a, FlashInfer for Hopper).

If you need to rebuild or use a different image, update the `VLLM_IMAGE` variable in `.env` (or the `image:` field in `docker-compose.yml`).

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Container definition with volume mounts, env vars, and entrypoint |
| `.env.example` | Template for environment configuration |
| `.env` | Your local environment configuration (git-ignored) |
| `start.sh` | Start server on both nodes |
| `stop.sh` | Stop server on both nodes |
| `.gitignore` | Files excluded from version control |

## Troubleshooting

### NCCL / InfiniBand

- Verify IB link state: `ibstat` (should show `LinkUp: true`)
- Test passwordless SSH: `ssh <WORKER_HOST> hostname`
- Check NCCL debug logs (set `NCCL_DEBUG=INFO` in compose env)

### Container fails to start

```bash
docker compose logs
```

### CUDA out of memory

Reduce `GPU_MEMORY_UTILIZATION` (e.g., `0.20`) or `MAX_MODEL_LEN` in `.env`.

### Timeout waiting for API

The model loads from HuggingFace on first run. Check logs:
```bash
docker logs qwen38-27b-vllm-2x-dgx-spark-vllm-1
```

## License

This repository's code is provided under the MIT License. The Qwen3.8 27B model weights are subject to [Qwen's license](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4).
