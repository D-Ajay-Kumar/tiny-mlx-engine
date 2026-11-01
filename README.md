# tiny-mlx-engine

A small LLM inference engine for Apple Silicon, built from scratch on [MLX](https://github.com/ml-explore/mlx), one commit at a time.

## Layout

- `src/tiny_mlx_engine/`: the engine (library code)
- `experiments/`: standalone scripts used to learn and measure

## Run

```bash
uv sync
uv run python experiments/01_arrays_and_shapes.py
```
