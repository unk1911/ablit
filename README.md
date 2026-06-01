# ablit

Launcher scripts and setup glue for running **abliterated / uncensored GGUF models** locally and on rented GPU boxes via [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

"Abliterated" models have had their refusal direction projected out of the residual stream (Labonne / FailSpy style), so they answer frank prompts without RLHF-style hedging. This repo is the thin wrapper that makes them easy to launch with sane flags for whatever hardware happens to be in front of you — an RTX 4090, a Grace-Hopper, a 1× H100 PCIe, or a 2× H100 SXM5 box.

## What's here

- `setup.sh` — clones/builds `llama.cpp` with CUDA, downloads a chosen model from HuggingFace into `ablit_models/`, and symlinks the build dir for the launchers.
- `run-*` scripts — one per model/hardware-class combination. Each one points at a specific GGUF, picks reasonable `-ngl` / `-c` / KV-cache / sampling flags, and auto-detects oneshot vs. interactive mode based on whether `-p` / `-f` is in the args:
  - `run-llama` — Llama 3.3 70B abliterated Q8, interactive defaults (c=8192).
  - `run-llama-big` — same model on 1× H100, `-ngl 70` + Q8 KV cache + FA, `-c 80000`.
  - `run-llama-xl` — same model on 2× H100, full offload, `-c 128000`, tensor-split 1,1.
  - `run-deckard`, `run-deckard-q8`, `run-deckard-xl`, `run-deckard-non-interactive` — Deckard 40B (Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic) variants. The `-xl` script uses Deckard's full 256K window.
  - `run-roughhouse` — sibling Qwen3.5-40B RoughHouse variant.
  - `run-qwen235-2507` — vanilla `Qwen3-235B-A22B-Instruct-2507` Q4_K_M on 2× H100 SXM5 with Q4 KV cache, `-c 230000`. Not abliterated; included because Qwen Instruct cooperates fully on the kind of analytical content abliterated models are usually picked for, and the long context wins.
  - `run-balanced`, `run-fast`, `run-max`, `run-quality` — preset profiles trading speed against output quality on smaller Gemma-heretic builds.

Output for oneshot runs is teed to `./full.log`. That file (and a few related prompt-output patterns) is in `.gitignore` since prompts and responses are often personal — **never `git add -A` in this repo**.

## Quickstart on a fresh GPU box

```bash
git clone https://github.com/unk1911/ablit ~/ablit
cd ~/ablit
./setup.sh --model llama70b     # builds llama.cpp + downloads model
./run-llama                     # interactive chat
./run-llama -p "your prompt"    # oneshot, output also written to full.log
./run-llama -f prompt.txt       # oneshot from file
```

For H100 / SXM5 deployment recipe (CUDA arch flags, KV-cache math, Xet downloads), see the playbook saved alongside this repo.

## Conventions

- Models live under `ablit_models/NN_short_name/...` where `NN` is a stable numeric prefix. Cloud boxes symlink real model dirs into that layout so the `run-*` scripts work unchanged.
- Each `run-*` is a single self-contained bash file — no shared library. New hardware class → add a new `-xl` / `-big` variant rather than parameterizing existing scripts.
- All scripts share the same oneshot-detection block: scanning `$@` for `-p` / `-f` / `--prompt` / `--file` flips into single-turn mode with `--log-disable` and tees to `full.log`; otherwise an interactive REPL is exec'd.

## Disclaimer

These models will produce content that aligned chat models refuse. Run them on prompts you own, and treat outputs as raw — they have no safety layer. The repo is a personal toolbox, not a product.
