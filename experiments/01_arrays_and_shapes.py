"""Lesson 1: arrays, shapes, and the operation that dominates LLM inference.

Run:  uv run python experiments/01_arrays_and_shapes.py
"""

import mlx.core as mx

mx.random.seed(0)

def section(title: str) -> None:
    print(f"\n=== {title} ===")

# 1) An array is a grid of numbers plus a shape.
section("1. arrays have shapes")
vector = mx.array([1.0, 2.0, 3.0])
matrix = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("vector:", vector.shape, vector.dtype)
print("matrix:", matrix.shape, matrix.dtype)
assert vector.shape == (3,)
assert matrix.shape == (2, 3)  # (rows, columns)

# 2) Embedding lookup: each token id picks one row from a big table.
section("2. token ids -> hidden states")
VOCAB_SIZE = 10  # real model (Qwen2.5-0.5B): 151_936
HIDDEN = 4  # real model (Qwen2.5-0.5B): 896
embedding_table = mx.random.normal((VOCAB_SIZE, HIDDEN))
token_ids = mx.array([3, 7, 1])  # pretend this is "the cat sat"
x = embedding_table[token_ids]  # picks rows 3, 7 and 1
print("token_ids:", token_ids.shape, "-> x:", x.shape)
assert x.shape == (3, HIDDEN)  # one row of HIDDEN numbers per token

# 3) A linear layer is one matmul: (tokens, in) @ (in, out) -> (tokens, out)
section("3. linear layer = matmul")
OUT = 8
weight = mx.random.normal((OUT, HIDDEN))  # stored (out, in), same as mlx.nn.Linear
y = x @ weight.T
print(f"x {x.shape} @ weight.T {weight.T.shape} -> y {y.shape}")
assert y.shape == (3, OUT)

# 4) The inner dimensions must match, or the matmul is rejected.
section("4. shapes must line up")
try:
    x @ weight  # (3, 4) @ (8, 4): inner dims are 4 and 8
except ValueError as err:
    print("expected error:", err)

# 5) Batching = one more dimension in front. Same weights serve every request.
section("5. batching adds a leading dimension")
batch = mx.stack([x, x])  # 2 requests, 3 tokens each
out = batch @ weight.T
print("batch:", batch.shape, "-> after linear:", out.shape)
assert out.shape == (2, 3, OUT)

# 6) dtype decides how many bytes each number costs.
section("6. dtype = bytes per number")
for dtype in (mx.float32, mx.float16, mx.bfloat16):
    a = mx.zeros((1024, 1024), dtype=dtype)
    print(f"{str(dtype):>14}: {a.itemsize} bytes/number, {a.nbytes / 1e6:.1f} MB")

# 7) Scale it up: a model's size in memory is just params x bytes per param.
section("7. real model sizes")
models = {"Qwen2.5-0.5B": 0.49e9, "Llama-3.1-8B": 8.0e9}
for name, n_params in models.items():
    for label, bytes_per_param in (("bf16", 2), ("4-bit", 0.5)):
        gb = n_params * bytes_per_param / 1e9
        print(f"{name:>14} {label:>5}: ~{gb:4.1f} GB of weights")
