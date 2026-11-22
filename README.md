# Performance Optimization of K-Means Using GPU Computing

Compare runtime as the number of clusters increases for NumPy CPU, scikit-learn
CPU, and one PyTorch GPU library baseline. No GPU performance claim is made until
measured on CUDA.

## Implementations

| Implementation | Assignment | Centroid update |
| --- | --- | --- |
| [NumPy CPU](implementations/vectorized_implementation.py) | Broadcast Euclidean distances | NumPy cluster means |
| [scikit-learn CPU](implementations/scikit_implementation.py) | Compiled scikit-learn Lloyd routine | Compiled scikit-learn Lloyd routine |
| [PyTorch GPU](implementations/pytorch_implementation.py) | Matrix multiplication using `torch.addmm`, followed by `argmin` | Tensor sums/counts via `index_add_` |

All three use Lloyd's algorithm. The former V0/V1 files have been consolidated into
`pytorch_implementation.py`. The GPU baseline uses matrix multiplication to compute
centroid squared norms minus twice the sample-centroid dot products. The sample
norm is omitted because it does not affect the nearest-centroid decision. Final
inertia uses actual squared residuals. Assignment stores an N × K score matrix.
This is a library-based CPU/GPU comparison, not a claim of a novel GPU algorithm.

NumPy and PyTorch retain previous centroids for empty clusters; scikit-learn retains
its relocation policy. CPU algorithms and default early stopping are unchanged.

## The benchmark

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m benchmarks.sweep_k --speedup
```

Install a CUDA-enabled PyTorch build appropriate to the NVIDIA machine. The default
command requires CUDA and fails explicitly if it is unavailable. It never substitutes
CPU PyTorch timings for GPU timings.

**Default dataset: Fashion-MNIST.** The benchmark uses all 70000 images and all
784 pixel features, converted to float32 and divided by 255 by the existing loader.
Class labels are unused. The first run downloads OpenML Fashion-MNIST v1 into
`datasets/`; subsequent runs reuse the cache. Dataset loading and preprocessing are
outside timing. No resampling or dimensionality reduction is applied.

Select the dataset explicitly with:

```bash
python -m benchmarks.sweep_k --dataset fashion-mnist --speedup
python -m benchmarks.sweep_k --dataset synthetic --speedup --output results/synthetic
```

The synthetic option retains N=100000, D=32, using a standard-normal float32 input
from `np.random.default_rng(42)`. A Fashion-MNIST download error stops the run; it
does not fall back to synthetic data. `environment.json` records the dataset,
preprocessing, and actual N/D so results from the two datasets stay distinguishable.

Progress bars show the overall number of implementation/K configurations and the
current K, implementation, warmup, timed run, and correctness-validation stage.
Each configuration has six fits: one warmup and five timed runs. Progress updates
happen outside the measured region; the 20-iteration algorithm loops are unchanged.
Use `--no-progress` to hide bars when saving terminal output.

Fixed protocol:

- `K = [2, 4, 8, 16, 32, 64, 128]`. N/D follow the selected dataset as described above.
- All implementations receive the same preprocessed float32 input. For each K,
  initial centroids are the same input rows selected with
  `np.random.RandomState(42).choice(..., replace=False)` for every implementation/run.
- Exactly **20 Lloyd assignment/update iterations**, then a final assignment and
  squared-residual inertia calculation. Early stopping is disabled explicitly.
- One full-size warmup for each implementation/K followed by **five timed runs**.
- Timing uses `perf_counter`, with CUDA synchronization immediately before starting
  and after fitting. Input transfer and initial GPU-centroid transfer are excluded.
  Timings include fitting setup/validation, centroid copies, all 20 updates, final
  assignment, and inertia computation. Independent correctness checks and result
  transfer back to CPU happen outside timing.
- TF32 is disabled and PyTorch float32 matrix multiplication precision is set to
  `highest`. CUDA atomic sums can still vary slightly between runs.
- CPU libraries and scikit-learn Lloyd use one thread by default. Use `--threads N`
  to choose another count; the setting is recorded.

**Why scikit-learn needs special handling:** `KMeans(tol=0, max_iter=20)` can still
stop when labels become unchanged. With `early_stopping=False`, our wrapper calls
scikit-learn's compiled `lloyd_iter_chunked_dense` routine exactly 20 times, followed
by one assignment-only call. This is a fixed-work wrapper around scikit-learn's
implementation, rather than the usual `KMeans.fit` stopping loop. It retains input
centering and empty-cluster relocation. This private API is isolated in one helper;
unsupported signatures fail rather than silently changing the workload. Tested
versions are listed below. The wrapper continues to return a fitted KMeans model.

### Outputs

The default output directory is `results/k_sweep/`:

| File | Contents |
| --- | --- |
| `runtime.csv` | All five raw runtime measurements per implementation/K |
| `environment.json` | Shape, seed, precision, thread count, OS/Python/dependency versions, and CUDA device/settings when used |
| `runtime_vs_k.png` | Median runtime; error bars show sample standard deviation across five runs |
| `speedup_vs_k.png` | Optional: median NumPy CPU runtime / median implementation runtime at each K |

Omit `--speedup` to generate only runtime vs K. No normalized-runtime, memory,
inertia, iteration-count, or phase-breakdown figures are produced. Inertia and
iteration counts are correctness checks, not plotted metrics. The earlier proposed
multi-figure patch was absent from this checkout, so no old figures needed deletion.

Use a fresh output directory for each experiment (`--output results/my_run`). The
runner rejects a nonempty directory to prevent mixing measurements. Partial CSV
results are checkpointed after each implementation/K; plots are written only after
the complete requested sweep passes validation. Generated results are ignored by Git.
To regenerate only the allowed plots from a completed run:

```bash
python -m benchmarks.plot_k results/k_sweep/runtime.csv --speedup
```

### Correctness checks

Every timed result is checked outside timing for exactly 20 updates, float32 finite
centroids, integer labels in range, and final nearest-centroid assignments. An
independent float64 calculation, in batches of up to 1024 rows (smaller for large
K/D to bound validation memory), checks selected squared
distances against the minimum (`rtol=2e-5`, `atol=2e-5`) and verifies reported inertia
at the same tolerance. Objective agreement with the first NumPy run at that K uses
`rtol=0.005`, `atol=1e-4`. This allows small floating-point trajectory differences;
it does not prove identical partitions. Failures stop the sweep. Empty-cluster
policies differ, so deliberately degenerate inputs need separate interpretation.
Unit tests also check centroid agreement on a controlled case and random input,
empty clusters, unchanged default stopping behavior, and rejection of wrong labels.

## Local verification and CUDA limitations

```bash
python -m unittest discover -s tests -v
python -m benchmarks.sweep_k --cpu-only --smoke --output results/fashion_smoke
python -m benchmarks.sweep_k --dataset synthetic --cpu-only --smoke --output results/synthetic_smoke
```

`--smoke` uses K=[2,4] and 256 rows. With Fashion-MNIST it selects the first
256 images and retains all 784 features; with synthetic data it uses N=256, D=4.
It still uses 20 iterations and five timed runs. Smoke results are tests, not
full-size measurements. Fashion-MNIST smoke runs still need the download/cache.
`--cpu-only` explicitly produces a partial comparison containing the two CPU curves.

Local verification uses macOS ARM64, Python 3.12, scikit-learn 1.5.1 and PyTorch 2.8.0.
Dataset selection is checked with a controlled Fashion-MNIST loader fixture, including
the default CLI selection, retained features, recorded shape, and no fallback on
download failure. Synthetic mode is checked without any Fashion-MNIST fetch.
The CUDA test is skipped when NVIDIA CUDA is unavailable.
The PyTorch baseline was tested on CPU for logical correctness. This does not validate
CUDA execution or GPU performance. Run the full test suite and benchmark on an
NVIDIA machine before reporting GPU results.

The earlier complete **synthetic** CPU sweep passed validation for all seven K values (70 timed runs).
Its local artifacts are in `results/k_sweep_cpu/`. At K=128, median runtimes were
12.6672 seconds for NumPy and 0.7915 seconds for scikit-learn on this machine, with
one CPU thread. These CPU-only measurements do not establish GPU speedup.

The NumPy broadcast implementation is preserved. On full Fashion-MNIST at K=128,
just one N × K × D float32 array is about 28 GB (26 GiB), and simultaneous
subtraction/squaring temporaries can require over 50 GB before other allocations.
Use a machine with sufficient RAM for the full comparison; `--smoke` is the small
functional check. Full Fashion-MNIST performance and CUDA execution have not been
validated locally.

## Layout

```text
implementations/              # CPU algorithms and one PyTorch baseline
benchmarks/sweep_k.py          # Shared data, initialization, timing, validation
benchmarks/plot_k.py           # Runtime and optional speedup figures only
tests/test_benchmark.py        # Correctness, iteration, output, and CUDA integration
requirements.txt              # CPU/PyTorch dependencies
```

The implementation scripts also remain runnable as Fashion-MNIST single-run
demonstrations. The K-sweep now defaults to that same dataset but applies the fixed
20-iteration, five-run protocol. Earlier single-run Fashion-MNIST measurements are not comparable to this
protocol and have been removed from the result presentation.
