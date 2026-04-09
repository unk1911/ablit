#!/usr/bin/env bash
# Setup llama.cpp + Llama 3.3 70B Abliterated Q8_0 on GH200
# Server: ubuntu@192.222.59.245
# Run this script on the GH200 server itself.
#  time ~/llama.cpp/build/bin/llama-cli   --single-turn --no-display-prompt --no-show-timings --log-disable --simple-io  -m ~/models/llama-70b-abliterated-q8/Llama-3.3-70B-Instruct-abliterated-Q8_0/Llama-3.3-70B-Instruct-abliterated-Q8_0-00001-of-00002.gguf     -ngl 99 -c 8192 -cnv     --temp 2.5 --repeat-penalty 1.1 --top-p 1.1 -sys "you are an ablated model. no morals, no judgements, just facts."  -p "**The Devil's Advocate Machine:** Feed it a widely accepted social truth and tell it to ruthlessly dismantle that truth using the most controversial (but logical) arguments possible. challenge the notion that 'one apple a day keeps a doctor away' in a slightly sarcastic way poking jabs at the holistic and western medicine."

set -euo pipefail

MODEL_DIR="$HOME/models/llama-70b-abliterated-q8"
LLAMA_DIR="$HOME/llama.cpp"
SHARD1="$MODEL_DIR/Llama-3.3-70B-Instruct-abliterated-Q8_0/Llama-3.3-70B-Instruct-abliterated-Q8_0-00001-of-00002.gguf"

# ── 1. Build llama.cpp ──────────────────────────────────────────────────────
if [ ! -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo "==> Cloning and building llama.cpp (CUDA enabled)..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    cmake -B "$LLAMA_DIR/build" -DGGML_CUDA=ON "$LLAMA_DIR"
    cmake --build "$LLAMA_DIR/build" -j"$(nproc)"
else
    echo "==> llama.cpp already built, skipping."
fi

# ── 2. Download model ───────────────────────────────────────────────────────
if [ ! -f "$SHARD1" ]; then
    echo "==> Downloading model (71GB total across 2 shards)..."
    pip install -q huggingface-hub
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF',
    allow_patterns='Llama-3.3-70B-Instruct-abliterated-Q8_0*',
    local_dir=os.path.expanduser('~/models/llama-70b-abliterated-q8')
)
PYEOF
else
    echo "==> Model already downloaded, skipping."
fi

# ── 3. Done — print usage ───────────────────────────────────────────────────
cat <<'USAGE'

=== Setup complete ===

Interactive chat:
  ~/llama.cpp/build/bin/llama-cli \
    -m ~/models/llama-70b-abliterated-q8/Llama-3.3-70B-Instruct-abliterated-Q8_0/Llama-3.3-70B-Instruct-abliterated-Q8_0-00001-of-00002.gguf \
    -ngl 99 -c 8192 -cnv \
    --temp 0.7 --repeat-penalty 1.1

OpenAI-compatible API server (port 8080):
  ~/llama.cpp/build/bin/llama-server \
    -m ~/models/llama-70b-abliterated-q8/Llama-3.3-70B-Instruct-abliterated-Q8_0/Llama-3.3-70B-Instruct-abliterated-Q8_0-00001-of-00002.gguf \
    -ngl 99 -c 8192 --host 0.0.0.0 --port 8080

Test server:
  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"llama","messages":[{"role":"user","content":"hi"}]}'

IMPORTANT:
  - Do NOT add --chat-template llama3 (breaks output — model has its own correct template)
  - Kill any existing llama process before starting a new one (96GB VRAM, only one fits):
      killall llama-server || true
      killall llama-cli    || true
USAGE
