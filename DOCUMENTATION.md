To run the CLI: 

```bash
uv run python3 -m tinygrad.llm --serve 9001 --max_context 8192 --model /home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf
```


Model to add will be the Q4_K_M models: `hf://LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf`
Local path: `/home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf`


uv run python3 -m tinygrad.llm --serve 9001 --max_context 8192 --model /home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-Q4_K_M.gguf


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