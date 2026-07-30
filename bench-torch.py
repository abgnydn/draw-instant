# /// script
# dependencies = ["torch"]
# ///
# draw.instant — PyTorch head-to-head for the boot fusion probe
#
# Mirrors bench.js exactly so the browser numbers and the PyTorch numbers are
# comparable on the SAME machine:
#
#   chain : y = sigmoid( tanh( relu( a*b + 1.0 ) - 0.5 ) * 2.0 )
#   size  : 1<<20 f32 elements (~4 MB per buffer)
#   timing: separate 5-run warm-up, then 25 timed runs, median, device fenced
#
# Eager PyTorch is the honest analogue of the probe's NAIVE path: one kernel
# launch per op, every intermediate materialized to global memory. Run with
# `uv run bench-torch.py` (PEP 723 pulls torch) or any env that has torch.
# Compare against the fused/naive ms the page prints on the same machine.

import statistics
import time

import torch

N = 1 << 20
WARMUP = 5
RUNS = 100      # iterations per timed batch (a 2x batch pairs with it)
REPEATS = 5     # batch pairs; the median across them is reported


def chain(a, b):
    t = a * b + 1.0
    t = torch.relu(t)
    t = t - 0.5
    t = torch.tanh(t)
    t = t * 2.0
    return torch.sigmoid(t)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_batch(device, a, b, count):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(count):
        chain(a, b)
    sync(device)
    return (time.perf_counter() - t0) * 1000


def bench(device):
    # Same method as bench.js: submit a batch of N and sync ONCE, repeat at 2N,
    # take the slope. per_iter = (t(2N) - t(N)) / N cancels the fixed sync cost
    # exactly. Syncing once per iteration (the obvious way) measures the sync,
    # not the work — and torch must be measured the same way as the WGSL path
    # or the comparison is meaningless.
    a = torch.rand(N, dtype=torch.float32, device=device) * 2 - 1
    b = torch.rand(N, dtype=torch.float32, device=device) * 2 - 1
    for _ in range(WARMUP):
        chain(a, b)
    sync(device)
    samples = []
    for _ in range(REPEATS):
        t_n = time_batch(device, a, b, RUNS)
        t_2n = time_batch(device, a, b, RUNS * 2)
        samples.append((t_2n - t_n) / RUNS)
    return statistics.median(samples)


if torch.backends.mps.is_available():
    dev = torch.device("mps")
elif torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

ms = bench(dev)
cpu_ms = bench(torch.device("cpu")) if dev.type != "cpu" else ms
print(f"torch {torch.__version__} · chain of 6 eager ops · {N} f32 · median of {RUNS}")
print(f"  {dev.type:4} : {ms:7.3f} ms")
if dev.type != "cpu":
    print(f"  cpu  : {cpu_ms:7.3f} ms")
print("compare: the page's boot probe prints fused/naive ms for the same chain")
print("on this machine's GPU via WebGPU.")
