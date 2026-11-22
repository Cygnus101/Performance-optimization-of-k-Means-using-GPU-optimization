# Performance Optimization of K-Means Using GPU Computing

Compare NumPy CPU, scikit-learn CPU, the PyTorch CUDA baseline, and independent
Numba CUDA V2.0–V2.5 implementations using **one complete-fit benchmark**.
No GPU performance claim is made until measured on NVIDIA CUDA.

## Run the unified sweep

From the repository root:

```sh
python -m pip install -r requirements.txt
python -m benchmarks.sweep_k --speedup
```

The default runs all 11 configurations: NumPy, sklearn, PyTorch, V2.0,
V2.1 at 128/256/512 threads per block, and V2.2–V2.5 at 256 threads per block.
V2.5 uses 32 shards. Every configuration shares the same timing and correctness
policy. `sweep_numba.py` and `requirements-numba.txt` have been consolidated into
`sweep_k.py` and `requirements.txt`; use the commands below instead of the old script.

| Selection | Command |
| --- | --- |
| Everything | `python -m benchmarks.sweep_k --speedup` |
| PyTorch + all Numba configurations | `python -m benchmarks.sweep_k --suite gpu --speedup --output results/gpu` |
| NumPy + sklearn + PyTorch | `python -m benchmarks.sweep_k --suite original --speedup --output results/original` |
| Numba only | `python -m benchmarks.sweep_k --suite numba --speedup --output results/numba` |
| NumPy + sklearn only | `python -m benchmarks.sweep_k --suite cpu --speedup --output results/cpu` |

`--cpu-only` remains an alias for `--suite cpu`. GPU suites require real NVIDIA
CUDA and never substitute CPU PyTorch or Apple MPS. Numba suites reject CUDA
simulator mode and require compute capability >= 6.0 for float64 atomic inertia.
PyTorch and Numba both use CUDA device 0. TF32 is disabled and PyTorch float32
matrix multiplication precision is set to `highest`.

## Dataset and fixed workload

The default uses the existing Fashion-MNIST loader: OpenML Fashion-MNIST v1,
**all 70,000 images, all 784 float32 pixel features, divided by 255**. Class labels
are unused; train/test images are combined because this is an unsupervised runtime
experiment. Downloads are cached in `datasets/`. A full-run shape check rejects
accidental subsets; download errors stop execution without a synthetic fallback.

- Default K is **`[2, 4, 8, 16, 32, 64]`**, excluding 128. Override it with
  `--k 2 8 32`. Values must be distinct positive integers no larger than N; the
  supplied order is used for execution and recorded in metadata. Custom lists may
  include other K values if you explicitly request them.
- All configurations receive the same CPU input and initial centroids selected
  with `np.random.RandomState(42).choice(N, K, replace=False)` for each K.
- Every fit executes **20 Lloyd assignment/update iterations**, followed by final
  assignment and squared-residual inertia. Early stopping is disabled.
- Each configuration/K has **one complete untimed warmup**, synchronized and
  validated, followed by **five separately timed, synchronized fits**. Warmup
  compiles the actual Numba kernels with the same shapes and argument types.
- The median of the five times is reported. Configuration order rotates across K.
- CPU libraries and sklearn use one thread by default; set `--threads N` to change it.
- Change only V2.1's block configurations with `--threads-per-block 64 128 256 512`.

Optional synthetic data uses N=100000, D=32 and
`np.random.default_rng(42).standard_normal(..., dtype=np.float32)`:

```sh
python -m benchmarks.sweep_k --dataset synthetic --suite gpu --k 2 4 8 16 --speedup --output results/synthetic_gpu
```

`--smoke` uses 256 rows and defaults to K=[2,4]; an explicit `--k` overrides that
K list. It retains 784 Fashion-MNIST features or uses
D=4 for synthetic data. It still executes 20 iterations and five runs. Smoke
results are functional checks, not full-size measurements. `--no-progress` hides
progress bars; progress updates happen outside timing.

## Complete-fit timing, including overhead

Every implementation starts a fit with CPU arrays and finishes with CPU labels,
centers and inertia. The same `measure_fit` function controls every measurement:

```text
Outside timer: load/normalize data, choose initial centroids, initialize contexts,
               execute the untimed full warmup

Synchronize GPU (where applicable)
START perf_counter
    Host fit preparation/checks and fresh clustering state
    Allocate device/working memory and transfer inputs/initial centers
    Prepare variant-specific layouts, including V2.2 transpose + extra transfer
    Execute 20 Lloyd updates
    Compute final assignment and inertia
    Copy labels, centers and inertia back to CPU
Synchronize GPU (where applicable)
STOP perf_counter

Outside timer: independent correctness checks, CSV writes, plots
```

For PyTorch, both `torch.as_tensor(..., device="cuda")` input transfers and
`.cpu().numpy()` result copies occur inside the timed adapter. Numba's complete
public `kmeans` call is timed directly. CPU implementations use the same host-input,
host-output boundary without GPU transfers. Working allocations, center copies,
host checks and layout overhead are charged to the implementation that needs them.
Dataset download/normalization, seeded row selection, reference execution and JIT
compilation are excluded. Allocator caches may persist after warmup; this measures
warmed complete-fit latency, not cold-start latency or only GPU kernel execution.

**Old measurements must be rerun.** The previous `sweep_k` PyTorch timings excluded
input transfers and labels/centers copied back to CPU. Do not combine those numbers
with the unified `complete_fit_v1` results. Old imported measurements below remain
historical data and are not rewritten or relabeled as complete-fit results.

## Implementations

| Implementation | Assignment | Centroid update |
| --- | --- | --- |
| [NumPy CPU](implementations/vectorized_implementation.py) | Broadcast Euclidean distances | NumPy cluster means |
| [scikit-learn CPU](implementations/scikit_implementation.py) | Compiled sklearn Lloyd routine | Compiled sklearn Lloyd routine |
| [PyTorch GPU](implementations/pytorch_implementation.py) | `torch.addmm` followed by `argmin` | Tensor sums/counts via `index_add_` |

The CPU, sklearn, PyTorch algorithms and their default early-stopping behavior
remain unchanged. PyTorch's sample norm is omitted from assignment scores because
it does not change the nearest-centroid choice; inertia uses actual residuals.
NumPy, PyTorch and Numba retain empty centers; sklearn retains its relocation policy.
The former V0/V1 files remain consolidated into `pytorch_implementation.py`.

**Why scikit-learn needs special handling:** `KMeans(tol=0, max_iter=20)` can still
stop when labels become unchanged. With `early_stopping=False`, our wrapper calls
scikit-learn's compiled `lloyd_iter_chunked_dense` routine exactly 20 times, followed
by one assignment-only call. This is a fixed-work wrapper around scikit-learn's
implementation, rather than the usual `KMeans.fit` stopping loop. It retains input
centering and empty-cluster relocation. This private API is isolated in one helper;
unsupported signatures fail rather than silently changing the workload. Tested
versions are listed below. The wrapper continues to return a fitted KMeans model.

## Independent Numba CUDA variants

All filenames use underscores, `v2_0_baseline.py` through `v2_5_reduction.py`.
The original baseline's contents and user comments are preserved. Its only change
was renaming `v2.0_baseline.py` to `v2_0_baseline.py`.

Each variant returns `(labels, centers, iterations, converged, inertia)` and accepts
the baseline arguments plus `threads_per_block=256`. V2.5 also accepts `shards=32`.
New variants share `common.run_kmeans` and reuse V2.0 kernels for stages they do not
replace. These are independent experiments against V2.0, not cumulative changes.
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

## Correctness checks

For every K, a separate CPU reference executes 20 float64 Lloyd steps from the
same initial centers, using bounded distance batches and float64 center means.
It retains empty centers, resolves ties by lowest index and computes final inertia
from direct residuals. It does not use any benchmarked implementation's result.
The reference runs once per K outside timing, under the configured CPU thread limit.

Every warmup and timed result must have exactly 20 iterations, no early-stopping
flag, finite float32 centers and integral labels in range. Independent float64
checks verify nearest-center assignments and reported inertia with
`rtol=2e-5, atol=2e-5`. Final inertia must agree with the reference within
`rtol=0.005, atol=0.0001`. Float32 atomic/reduction order can change near ties and
later Lloyd trajectories; the 0.5% objective threshold is a regression check,
not proof of identical partitions. Failures stop the sweep; do not silently relax
tolerances to obtain plots. Controlled fixtures additionally compare centers at
tight tolerances. Sklearn's different empty-cluster policy can affect degenerate cases.

## Outputs

Default directory: `results/k_sweep/`. Use separate `--output` folders for different
suites or machines.

| File | Contents |
| --- | --- |
| `runtime.csv` | Five individual complete-fit times and validated/reference inertia per configuration/K |
| `medians.csv` | Median runtime for every configuration/K |
| `environment.json` | Dataset/shape, seed, suite, configurations, versions, GPU/precision, timing protocol, validation policy and completion status |
| `runtime_vs_k.png` | Median complete-fit time vs K, logarithmic axes; error bars show run sample standard deviation |
| `speedup_vs_k.png` | Optional ratio of baseline median / configuration median |

Speedup uses **NumPy CPU** when the suite includes CPUs, otherwise **V2.0 baseline**.
The denominator is stated on the plot and in metadata. A value above 1 means faster
than that baseline. The GPU-only suite therefore compares PyTorch and Numba against
the same V2.0 denominator. Only runtime and optional speedup figures are generated.

Once device/data setup succeeds, reruns clear these five known artifacts and
preserve unrelated files. Every validated timed run checkpoints the CSV. Metadata
is `running` until completion, `failed` on an exception, and `complete` only after
all runs, validation and final outputs succeed. Failed runs retain partial CSV data
but remove final medians/plots. To regenerate plots from a completed raw CSV:

```sh
python -m benchmarks.plot_k results/k_sweep/runtime.csv --speedup
```

## Colab / NVIDIA execution

Copy this working tree to Colab, or commit and push it before cloning. A clone of
the public repository will not include unpushed local changes. Select
**Runtime > Change runtime type > GPU**, then run:

```python
%cd /content/Performance-optimization-of-k-Means-using-GPU-optimization
%pip install -r requirements.txt
```

The single requirements file includes CPU, PyTorch and Numba dependencies. On
Linux it also installs `numba-cuda[cu12]` toolkit libraries; the GPU runtime must
provide a compatible NVIDIA driver and CUDA-enabled PyTorch. Restart the session
if PyTorch or Numba was imported before installation. See the official
[Numba CUDA installation instructions](https://nvidia.github.io/numba-cuda/user/installation.html).

```python
!nvidia-smi
!env -u NUMBA_ENABLE_CUDASIM python -c "import torch; from implementations.numba_cuda.common import require_cuda; print(require_cuda().name); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
!env -u NUMBA_ENABLE_CUDASIM NUMBA_TEST_REQUIRE_GPU=1 python -m unittest discover -s tests -v
!env -u NUMBA_ENABLE_CUDASIM python -m benchmarks.sweep_k --suite gpu --dataset synthetic --smoke --speedup --output results/gpu_smoke
!env -u NUMBA_ENABLE_CUDASIM python -m benchmarks.sweep_k --speedup --output results/unified
```

For the full GPU comparison without CPU algorithms, replace the final command's
suite with `--suite gpu`. Keep `environment.json` alongside measurements.

**Memory matters for `--suite all`:** the unchanged NumPy broadcast implementation
creates an N x K x D temporary. At full Fashion-MNIST K=64, one float32 temporary
is about 14 GB (13.1 GiB), and overlapping temporaries can require over 28 GB before
other allocations. A normal Colab host may run out of RAM. Use a sufficiently large
host for the all-implementation comparison, or `--suite gpu` for the full PyTorch +
Numba comparison. The script never silently omits an implementation or shrinks data.

## Local checks and remaining GPU validation

```sh
python -m unittest discover -s tests -v
NUMBA_ENABLE_CUDASIM=1 python -m unittest discover -s tests -p 'test_numba_cuda.py' -v
python -m benchmarks.sweep_k --suite cpu --dataset synthetic --smoke --speedup --output results/cpu_smoke
```

Local checks exercise the independent reference, invalid-result rejection, CPU
sweeps/output cleanup, complete-fit timer boundaries including transfers, unified
adapter orchestration, and unchanged CPU/PyTorch logical results. Numba simulation
covers non-warp kernels, fixed iterations, empty centers, early stopping and partial
image/feature/centroid tiles. It cannot compile device code or implement warp
shuffle; V2.3 explicitly rejects simulator execution. See the official
[simulator limitations](https://nvidia.github.io/numba-cuda/user/simulator.html).

**Still requires NVIDIA validation:** actual device compilation, warp execution,
atomics/barriers, all-K 784-feature kernel checks, real unified PyTorch+Numba smoke
execution, full Fashion-MNIST tolerances/memory use, and the complete five-run sweep.
`NUMBA_TEST_REQUIRE_GPU=1` makes absent real CUDA an error instead of skipped tests.
The hardware suite checks all six K values and eight Numba configurations on
137 images of 784 features and also exercises the unified GPU smoke runner.
No new GPU timings or speedup claims have been generated on this Mac.

## Historical measurements

The imported medians in this section used the old transfer-excluding timing policy.
They are preserved for provenance and must not be mixed with `complete_fit_v1`.

### Imported Fashion-MNIST console medians

`data/benchmarks/fashion_mnist_console_medians.csv` was extracted from the uploaded
`Pasted markdown(1).md` log of `python -m benchmarks.sweep_k --speedup`.
The source reports Fashion-MNIST, N=70000, D=784, smoke=False. Only complete
`K=... implementation: median ...s` lines count as results; progress-bar elapsed
times and rates are excluded. The K=64 PyTorch GPU median (0.157666s) was supplied
in a follow-up message. All 18 measured runtime strings retain the exact
six decimal places printed in the log. No individual run timings or uncertainty
estimates can be recovered from these summaries.

The CSV has 21 rows covering the seven configured K values and three implementations.
Missing results have `status=missing` and empty numeric fields for all
implementations at K=128. A completed 6/6 progress bar is insufficient
without the final median line. Missing entries are placeholders, not measurements.

For the same K, `speedup_vs_numpy_cpu` is NumPy CPU median / row median;
`speedup_vs_pytorch_gpu` is PyTorch GPU median / row median. Both are dimensionless:
greater than 1 means the row implementation is faster than the named baseline.
Ratios are derived from the printed medians and rounded to 12 decimal places;
they do not recover the runner's unprinted precision. If either median is missing,
the corresponding speedup is blank.

The CSV is already saved in this checkout. In another checkout, copy it to
`data/benchmarks/fashion_mnist_console_medians.csv`. From the repository root,
load it using the Python standard library:

```python
import csv
from pathlib import Path

path = Path("data/benchmarks/fashion_mnist_console_medians.csv")
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
completed = [row for row in rows if row["status"] == "completed"]
# Convert a runtime when needed; keeping strings preserves logged decimal digits.
seconds = float(completed[0]["median_runtime_seconds"])
```

Plot runtime versus K without rerunning the benchmark:

```bash
python -m benchmarks.plot_medians data/benchmarks/fashion_mnist_console_medians.csv
```

This writes `fashion_mnist_console_medians_runtime_vs_k.png` beside the CSV.
Both axes use logarithmic scales so the CPU and GPU curves remain visible.
Missing values remain gaps; K=128 stays labeled with no measured points.
There are no error bars because the log provides only medians. Use this dedicated
plotter for the imported summary: `benchmarks.plot_k` expects per-run `seconds`
and calculates run standard deviations, so the schemas are intentionally distinct.

## Layout

```text
implementations/                 # Existing CPU/PyTorch algorithms
implementations/numba_cuda/      # V2.0 baseline, V2.1–V2.5, shared runner
benchmarks/sweep_k.py             # Single unified complete-fit sweep
benchmarks/validation.py          # Independent float64 CPU reference/checks
benchmarks/numba_configs.py       # Numba configuration registry
benchmarks/plot_k.py              # Runtime and optional speedup plots
benchmarks/plot_medians.py        # Historical imported-summary plotter
tests/                           # Algorithm, runner, timing and CUDA checks
requirements.txt                 # Merged dependencies
```
