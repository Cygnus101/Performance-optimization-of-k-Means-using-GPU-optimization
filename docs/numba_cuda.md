# Independent Numba CUDA k-means variants

All version filenames use underscores: `v2_0_baseline.py` through
`v2_5_reduction.py`. The original baseline was renamed from `v2.0_baseline.py`;
its contents are unchanged. CPU, sklearn, vectorized, PyTorch implementations and
`benchmarks/sweep_k.py` are unchanged by this work.

Each variant returns `(labels, centers, iterations, converged, inertia)` and accepts
the baseline arguments plus `threads_per_block=256`. V2.5 also accepts `shards=32`.
New variants share `common.run_kmeans` and the original baseline's reset, update,
accumulation and inertia kernels wherever those stages are unchanged. They are
independent experiments relative to V2.0, not a cumulative optimization pipeline.
No Numba implementation imports or uses PyTorch.

| File under `implementations/numba_cuda/` | Change relative to V2.0 | Tradeoff |
| --- | --- | --- |
| `v2_0_baseline.py` | One thread/image assignment; global float32 atomic accumulation | Original comparison point; 256 threads/block |
| `v2_1_block_tuning.py` | Configurable launch size for every kernel | Sweep defaults to 128, 256, 512; no universal best block size |
| `v2_2_coalescing.py` | Assignment reads a contiguous `(D,N)` copy: adjacent image threads read adjacent addresses at each feature | Keeps `(N,D)` for accumulation/inertia; extra memory, transpose and transfer count in timing |
| `v2_3_warp.py` | One warp/image; lanes read consecutive features, then shuffle-reduce each centroid distance | Full 32-lane warps, ascending centroid tie-breaking in lane 0; padded feature tails; changes addition order |
| `v2_4_shared_memory.py` | One image/thread; reuse 4-centroid x 128-feature tiles across the block | 2 KiB shared memory/block plus four local distances/thread; all threads reach barriers even for partial image blocks; synchronization costs can outweigh reuse |
| `v2_5_reduction.py` | Blocks atomically accumulate into one of 32 `(K,D)` shards; a second kernel merges shards without atomics | Same number of sample atomics, spread over independent destinations; extra clear/merge work and `shards*K*(D*4+4)` bytes of scratch |

The variants keep empty centroids, use float32 features/centers/sums, and use the
baseline float64 atomic accumulator for final inertia. Compute capability >= 6.0
is required. All distances use direct squared differences on the GPU; no fast-math
option is enabled. Default early stopping is supported, but the sweep disables it.
The baseline's final assignment runs again after the last center update.

## Dedicated sweep and timing

Run from the repository root:

```sh
python -m benchmarks.sweep_numba --output results/numba_sweep --speedup
```

This script does not invoke the existing sweep. It calls the existing
`common.load_fashion_mnist`, which imports the same vectorized loader used elsewhere:
OpenML Fashion-MNIST v1, all 70,000 images (train + test), 784 float32 features,
pixels divided by 255, labels unused. A shape check prevents an accidental subset
or different dataset. No synthetic or reduced-data mode exists in the sweep.

Protocol:

- K is exactly `{2,4,8,16,32,64}`. No K=128.
- Every fit executes exactly 20 Lloyd updates, then final assignment and inertia.
- For each K, select the same initial centers with
  `np.random.RandomState(42).choice(N, K, replace=False)` for every implementation,
  warmup, timed run, and reference. Center inputs are never updated in place.
- Before each configuration's five measurements, run one full untimed 20-iteration
  fit on the same shapes and argument types, synchronize, and validate it. This
  forces that configuration's required kernels to compile outside the timer.
- Immediately before starting `perf_counter`, call `cuda.synchronize()`. Time the
  **complete public `kmeans` call** and a final `cuda.synchronize()` before stopping
  the clock. Include host float32 preparation/checks, device allocations, H2D,
  layout conversion/extra H2D for V2.2, all kernels, and result D2H copies. Device
  allocator caching may persist after warmup; this is steady-state complete-fit
  latency, not cold-start latency or kernel-only timing.
- Exclude data download/preprocessing, initial-center selection, JIT warmup,
  reference execution, validation, CSV writes and plotting. All configurations use
  the same boundaries; new variants also include their host argument checks.
- Rotate configuration order across K to reduce systematic temperature/order bias.
  This does not eliminate hardware noise. Five separate fits are saved, then the
  median is plotted. Do not compare these numbers with the other sweep's timing
  protocol, which excludes input transfer.

The default is eight configurations per K: V2.0, V2.1 at 128/256/512, and V2.2–V2.5
at 256 threads/block with 32 shards for V2.5. To change the V2.1 configurations:

```sh
python -m benchmarks.sweep_numba --threads-per-block 64 128 256 512 --speedup
```

Outputs in the dedicated directory:

- `runtime.csv`: all five individual synchronized times plus validated objectives.
- `medians.csv` and `runtime_vs_k.png`: median complete fit runtime versus K.
- `speedup_vs_k.png`: optional ratio `V2.0 median / variant median`.
- `environment.json`: GPU, compute capability, runtime/package versions, dataset,
  seed, configurations and exact timing/validation policy; status becomes
  `complete` only after all measurements pass. An interrupted/failed run may have
  partial CSV data and status `running`; it does not produce final plots.

The full sweep can take substantial time: assignment does O(N*K*D) work, and the
float64 CPU reference and per-run correctness checks also take time outside timing.
These variants are experiments; no speedup is assumed or promised.

## Correctness policy

Before measuring any implementation at each K, a separate CPU implementation runs
20 float64 Lloyd steps from the same centers. It uses bounded distance batches and
float64 center means, retains empty clusters, resolves ties by lowest index, and
computes the final objective from direct residuals. It does not reuse CUDA labels,
CUDA centroids, sklearn fitting, or any benchmarked implementation's results.

Every warmup and timed result must have finite float32 centers, valid integral
labels, exactly 20 iterations and no convergence flag. Independently recomputed
float64 distances must show each reported label is a nearest center within
`rtol=2e-5, atol=2e-5`; reported inertia must match direct residual inertia to the
same tolerances. Final inertia must match the independent float64 reference within
`rtol=0.005, atol=0.0001`. Float32 atomics and warp/shard reductions change summation
order; near ties can change later Lloyd trajectories, so full-data labels/centers
are not required to be bit-identical. The 0.5% objective check is a regression
threshold, not proof of an identical trajectory. A failure stops the sweep; do not
silently relax tolerances to obtain a plot. Small stable fixtures additionally
compare centers directly at tight tolerances.

## Colab: exact GPU commands

First copy/upload this working tree (including the new files) to Colab, or commit
and push it before cloning. Cloning the current public repository alone will not
include these local changes. In Colab select **Runtime > Change runtime type > GPU**.
Then, from a notebook cell (adjust the first path to your checkout):

```python
%cd /content/Performance-optimization-of-k-Means-using-GPU-optimization
%pip install -r requirements-numba.txt
%pip install "numba-cuda[cu12]"
```

Restart the Colab session after installation if Numba was already imported. The
CUDA 12 extra installs the toolkit libraries needed by Numba; an NVIDIA driver
must already be supplied by the GPU runtime. See the official
[Numba CUDA installation instructions](https://nvidia.github.io/numba-cuda/user/installation.html).
Then run:

```python
!nvidia-smi
!env -u NUMBA_ENABLE_CUDASIM python -c "from implementations.numba_cuda.common import require_cuda; d = require_cuda(); print(d.name, d.compute_capability)"
!env -u NUMBA_ENABLE_CUDASIM NUMBA_TEST_REQUIRE_GPU=1 python -m unittest discover -s tests -p 'test_numba_cuda.py' -v
!env -u NUMBA_ENABLE_CUDASIM python -m benchmarks.sweep_numba --output results/numba_sweep --speedup
```

The suite fails immediately if `NUMBA_TEST_REQUIRE_GPU=1` and real CUDA is absent.
The hardware test exercises all six K values, all eight configurations, 20 updates,
137 images of 784 features, partial image blocks and warp reductions. The full
sweep then validates all 70,000 images on every run. Record `environment.json`
with results. These are actual `@cuda.jit` kernel launches, not CPU substitutes or
PyTorch operations. Apple MPS is not a CUDA backend. The sweep rejects simulator
mode and absent NVIDIA CUDA before loading data or creating result files.

## Local checks and remaining validation

On a Mac with Numba installed:

```sh
python -m unittest discover -s tests -p 'test_numba_cuda.py' -v
NUMBA_ENABLE_CUDASIM=1 python -m unittest discover -s tests -p 'test_numba_cuda.py' -v
```

Local checks cover the CPU reference, validator rejection of bad assignments and
inertia, configuration/median/plot generation, fixed iteration counts, retained
empty centers, early stopping, unmodified initial centers, non-warp kernels and
partial feature/centroid/image tiles under simulation. The simulator deliberately
rejects V2.3: it does not implement warp shuffle primitives. See the official
[simulator limitations](https://nvidia.github.io/numba-cuda/user/simulator.html).

**Still requires Colab/NVIDIA validation:** device compilation of all kernels,
real warp shuffle execution, hardware synchronization and float64 atomics, all-K
784-feature hardware tests, memory use, full Fashion-MNIST reference tolerances,
and the complete warmed five-run sweep/plots. Simulator tests cannot establish
CUDA device compilation correctness, race freedom or performance. No GPU timings
or speedup claims have been generated locally.

Local verification on 2026-11-23 (Python 3.12, NumPy 1.26.4, Numba 0.60.0):
the Numba suite passed 8 tests under simulation with the one hardware test
skipped; without simulation it passed 4 CPU tests with 5 GPU tests skipped.
The existing benchmark suite passed 12 tests with its CUDA test skipped.
Syntax/import checks passed, and the real sweep correctly refused to run
without CUDA before creating any output directory.
