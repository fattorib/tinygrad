To run the CLI: 

```bash
JITBEAM=1 uv run python3 -m tinygrad.llm --serve 9001 --max_context 262144 --model /home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf
```


Model to add will be the Q4_K_M models: `hf://LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf`
Local path: `/home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf`


# Typechecking and linting:

```bash
uv run python -m ruff check . --fix
uv run python -m mypy tinygrad/llm
```

# Model

LFM arch is here: `https://www.liquid.ai/blog/lfm3`

- Conv block is very simple, really not that much goes into it, window size of 7 instead of 3
- Attn is standard GQA + gated
- FFN block is Swiglu
- We alternate between full and windowed attention
- Weird partial RoPE applied to global attention, this isn't used for SWA


# Benchmarking

```
export MPATH=/home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf

JITBEAM=0 uv run python3 -m tinygrad.llm --max_context 262144 --model $MPATH --warmup --benchmark 2500,128
```


Initial Results
(2500, 128)
```
JITBEAM=0
prefill  2500: 4314.07 ms,   579.50 tok/s,  102.50 GB/s, 442202/8970 MB
...
decode    127:  25.09 ms,    39.86 tok/s,   58.40 GB/s, 1464/9006 MB
```

```
JITBEAM=1
prefill  2500: 4548.64 ms,   549.62 tok/s,   89.31 GB/s, 406236/8970 MB
...
decode    127:  23.67 ms,    42.24 tok/s,   59.87 GB/s, 1417/9007 MB
```

```JITBEAM=2
prefill  2500: 3602.34 ms,   693.99 tok/s,  118.06 GB/s, 425286/9203 MB
...
decode    127:  22.45 ms,    44.55 tok/s,   61.32 GB/s, 1376/9240 MB
```