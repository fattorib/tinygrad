To run the CLI: 

```bash
uv run python3 -m tinygrad.llm --serve 9001 --max_context 8192 --model llama3.2:1b
```


Model to add will be the Q4_K_M models: `hf://LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q4_K_M.gguf`
Local path: `/home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots/e7caca5d835a3901a8e0d63e94009429bafafdfc/LFM2.5-2.6B-Q4_K_M.gguf`


uv run python3 -m tinygrad.llm --serve 9001 --max_context 8192 --model /home/benjamin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots/e7caca5d835a3901a8e0d63e94009429bafafdfc/LFM2.5-2.6B-Q4_K_M.gguf


# Typechecking and linting:

```bash
uv run python -m ruff check .
uv run python -m mypy tinygrad/llm
```

# Model

LFM arch is here: `https://arxiv.org/pdf/2511.23404`

- Conv block is very simple, really not that much goes into it
- Attn is standard GQA, ungated
- FFN block is Swiglu