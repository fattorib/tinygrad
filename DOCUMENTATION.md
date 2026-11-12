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

JITBEAM=1 uv run python3 -m tinygrad.llm --max_context 262144 --model $MPATH --warmup --benchmark 2500,128
```


1. Initial Results
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


2. Flash Attention on non-SWA Layers

```JITBEAM=1
prefill  2500: 3730.92 ms,   670.08 tok/s,   70.82 GB/s, 264212/8492 MB
...
decode    127:  18.21 ms,    54.91 tok/s,   71.23 GB/s, 1297/8528 MB
```

3. Flash Attention all all layers -> this is not a valid kernel and is broken

```JITBEAM=1
prefill  2500: 3400.21 ms,   735.25 tok/s,   37.21 GB/s, 126514/7319 MB
...
decode    127:  13.75 ms,    72.74 tok/s,   86.95 GB/s, 1195/7322 MB
```


4. SWA Kernel Modifications

```JITBEAM=1
prefill  2500: 3949.75 ms,   632.95 tok/s,   66.89 GB/s, 264212/8457 MB  --
decode    127:  15.48 ms,    64.59 tok/s,   75.40 GB/s, 1167/8460 MB
```

5. Custom Flash SWA for Prefill too:
Why did this increase decode speed?
```JITBEAM=1
prefill  2500: 3257.71 ms,   767.41 tok/s,   38.84 GB/s, 126514/7319 MB
decode    127:   9.78 ms,   102.21 tok/s,  119.33 GB/s, 1167/7322 MB
```

6. Better SWA for Prefill:
```JITBEAM=1
prefill  2500: 3269.78 ms,   764.58 tok/s,   37.70 GB/s, 123257/7553 MB
decode    127:   9.51 ms,   105.12 tok/s,  122.72 GB/s, 1167/7556 MB
```


7. JIT the CPU copybacks -> one less HCQ submit & link which adds overhead of ~1ms
```JITBEAM=1
prefill  2500: 3787.65 ms,   660.04 tok/s,   41.53 GB/s, 157307/7320 MB  --
decode    127:   7.81 ms,   128.05 tok/s,  204.67 GB/s, 1598/7323 MB  --  1203 1204 1205 1206 1207 1208 1209
```


Issues now:
1. Cache just gets randomly invalidated sometimes (when it should persist):

```
JITBEAM=1 uv run python3 -m tinygrad.llm --max_context 262144 --model $MPATH --warmup --serve 9001
```

```bash
*** started server on http://127.0.0.1:9001 at 65.98 s                                                                                                                                                          
/v1/chat/completions  --  prep:    7 ms  --  in:    0 + 1866  --  prefill: 744 tok/s  --  gen:  37 tok/s  --  out:   65  --  total:  4.29s
/v1/chat/completions  --  prep:   11 ms  --  in: 1931 +   31  --  prefill:  69 tok/s  --  gen:  96 tok/s  --  out:   40  --  total:  0.87s
/v1/chat/completions  --  prep:   20 ms  --  in: 2002 +  694  --  prefill: 639 tok/s  --  gen: 102 tok/s  --  out:  448  --  total:  5.48s
/v1/chat/completions  --  prep:   20 ms  --  in: 3144 +   23  --  prefill:  52 tok/s  --  gen:  98 tok/s  --  out:   84  --  total:  1.30s
/v1/chat/completions  --  prep:   22 ms  --  in:    0 + 3302  --  prefill: 817 tok/s  --  gen: 101 tok/s  --  out:   45  --  total:  4.49s -> weird invalidation here
/v1/chat/completions  --  prep:   53 ms  --  in: 3347 + 6528  --  prefill: 841 tok/s  --  gen:  98 tok/s  --  out:  588  --  total: 13.78s
/v1/chat/completions  --  prep:   57 ms  --  in:    0 +10480  --   -> weird invalidation here
```
This is the conversation: `file:///home/benjamin/tinygrad/pi-session-2026-11-03T13-37-06-691Z_01a1a1a0-b683-7a8e-96a2-604a4b696e66.html`


2. preserve_thinking should be false



## Top-k

Baseline:

```JITBEAM=1
prefill  2500: 3369.49 ms,   741.95 tok/s,   36.58 GB/s, 123257/7319 MB  --
decode    127:  10.10 ms,    107.33 tok/s,  125.30 GB/s, 1167/7322 MB  --  1203 1204 1205 1206 1207 1208 1209
```

Stock tinygrad top-k:

```JITBEAM=1
prefill  2500: 3659.14 ms,   683.22 tok/s,   39.32 GB/s, 143890/7367 MB  --
decode    127:  21.46 ms,    46.60 tok/s,   66.57 GB/s, 1428/7386 MB  --  1203 1204 1205 1206 1207 1208 1209
```


Faster tinygrad top-k:

```JITBEAM=1
prefill  2500: 3310.61 ms,   755.15 tok/s,   47.52 GB/s, 157307/7320 MB  -- 
decode    127:   8.26 ms,   121.07 tok/s,  193.52 GB/s, 1598/7323 MB  --  1203 1204 1205 1206 1207 1208 1209
```


# Can we drop .contiguous()?
- Removed in SwiGLU

```JITBEAM=1
prefill  2500: 3239.65 ms,   771.69 tok/s,   48.56 GB/s, 157307/7354 MB  --
decode    127:   8.79 ms,   113.82 tok/s,  181.93 GB/s, 1598/7357 MB  --  1203 1204 1205 1206 1207 1208 1209
```