#!/usr/bin/env bash
# serve_gemma.sh
# Serves Gemma 2B with vLLM on a laptop 5070 (8 GB VRAM)
# while another GPU process is running.

set -e

# ---- 1. Memory fragmentation helper ----
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- 2. Activate the venv ----
source .venv/bin/activate

# ---- 3. Show GPU state ----
echo "----- GPU state before launch -----"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
echo "-----------------------------------"

# ---- 4. Launch vLLM ----
# You have ~6.5 GB free. Gemma 2B fp8 is ~2.5 GB weights + ~1 GB overhead,
# leaving ~3 GB for KV cache. Plenty for max-num-seqs 1 at 1024 context.
#
# --gpu-memory-utilization 0.75 means vLLM gets 75% of the 8 GB TOTAL,
# which is ~6.1 GB. Fits inside your 6.5 GB free with a small safety margin.

vllm serve ./gemma4_2b \
  --quantization fp8 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 1 \
  --enforce-eager \
  --dtype auto
