# Ablit Models Layout

This folder centralizes all local GGUF models used in this project.

Ordering convention:
- `01` smallest / fastest
- higher numbers are heavier / slower but usually higher fidelity

## Model Map

1. `01_gemma4_e4b_heretic_q4_k_m`
   - File: `gemma-4-E4B-it-heretic-Q4_K_M.gguf`
   - Approx size: 5.0G
   - Best for fast chat and iteration

2. `02_gemma4_26b_a4b_heretic_q4_k_m`
   - File: `coder3101_gemma_4_26b_a4b_it_heretic-Q4_K_M.gguf`
   - Approx size: 16G
   - Good quality/speed balance

3. `03_gemma4_31b_heretic_q4_k_m`
   - Files:
     - `gemma-4-31b-it-heretic-Q4_K_M.gguf`
     - `mmproj-coder3101_gemma_4_31b_it_heretic-Q4_K_M.gguf`
   - Approx sizes: 18G + 628M
   - 31B class at moderate quant

4. `04_gemma4_31b_heretic_q6_k`
   - Files:
     - `coder3101_gemma_4_31b_it_heretic-Q6_K.gguf`
     - `mmproj-coder3101_gemma_4_31b_it_heretic-Q6_K.gguf`
   - Approx sizes: 24G + 771M
   - Higher quality than Q4, still practical

5. `05_gemma4_31b_heretic_q8_0`
   - Files:
     - `coder3101_gemma_4_31b_it_heretic-Q8_0.gguf`
     - `mmproj-coder3101_gemma_4_31b_it_heretic-Q8_0.gguf`
   - Approx sizes: 31G + 1010M
   - Heaviest / best fidelity, slowest on 4090

## Example llama.cpp commands

Fastest:
```bash
~/ablit/llama.cpp/build-cuda/bin/llama-cli -m ~/ablit/ablit_models/01_gemma4_e4b_heretic_q4_k_m/gemma-4-E4B-it-heretic-Q4_K_M.gguf -ngl auto -c 2048 -n 256 -st --simple-io --reasoning off -p "Hello"
```

Balanced:
```bash
~/ablit/llama.cpp/build-cuda/bin/llama-cli -m ~/ablit/ablit_models/02_gemma4_26b_a4b_heretic_q4_k_m/coder3101_gemma_4_26b_a4b_it_heretic-Q4_K_M.gguf -ngl auto -c 2048 -n 256 -st --simple-io --reasoning off -p "Hello"
```

Quality mode:
```bash
~/ablit/llama.cpp/build-cuda/bin/llama-cli -m ~/ablit/ablit_models/04_gemma4_31b_heretic_q6_k/coder3101_gemma_4_31b_it_heretic-Q6_K.gguf -ngl auto -c 1024 -n 256 -st --simple-io --reasoning off -p "Hello"
```

## Quick wrappers

Use these shortcuts from `~/ablit`:

```bash
./run-fast -p "Hello"
./run-balanced -p "Hello"
./run-quality -p "Hello"
./run-max -p "Hello"
```

You can pass any additional `llama-cli` flags through the wrapper.
