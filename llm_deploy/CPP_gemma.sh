#!/bin/bash
~/Menochat/llm_deploy/llama.cpp/llama.cpp/build/bin/llama-server \
  -m ~/Menochat/llm_deploy/Q8_gguf/afi_gemma4_e2b_merged-Q8_0.gguf \
  --host 0.0.0.0 \
  --reasoning-format auto\
  --port 8000 \
  --alias meno \
  -c 4096 \
  -ngl 99 \
  --parallel 1 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -b 256 \
  -ub 256
