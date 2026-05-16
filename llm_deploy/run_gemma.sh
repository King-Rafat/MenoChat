source .venv/bin/activate && vllm serve ./gemma4_2b \
  --quantization fp8 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 2 \
  --enforce-eager \
  --enable-chunked-prefill
