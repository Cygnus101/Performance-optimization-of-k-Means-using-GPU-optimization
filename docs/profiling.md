# Comparative stage profiling

Run from the repository root with the existing `requirements.txt` dependencies:

```sh
python -m benchmarks.profile_stages --list-versions
# All 12 current configurations, complete default workload:
python -m benchmarks.profile_stages --output results/profile_all
# Kaggle Tesla T4, single device; CPU baselines omitted explicitly:
env -u NUMBA_ENABLE_CUDASIM python -m benchmarks.profile_stages \
  --backend gpu --device 0 --k 2 4 8 16 32 64 --runs 5 \
  --output /kaggle/working/results/profile_gpu
# Functional GPU check before the full sweep:
env -u NUMBA_ENABLE_CUDASIM NUMBA_TEST_REQUIRE_GPU=1 \
  python -m unittest discover -s tests -p test_profile_stages.py -v
# Explicit subset, including V2.1 launch configurations:
python -m benchmarks.profile_stages --versions pytorch v2.0 v2.1:128 v2.1:256 \
  v2.1:512 v2.3 v2.5 v2.6 --k 8 32 --runs 5 --output results/profile_subset
# Local CPU functional check, no download or NVIDIA device needed:
python -m benchmarks.profile_stages --backend cpu --dataset synthetic --smoke \
  --runs 5 --output results/profile_cpu_smoke
```

`--backend` selects all/cpu/torch/numba/gpu. `--versions` selects current version
IDs, optionally narrowed by backend; incompatible selections fail. `--device`
selects the same NVIDIA index for both libraries. CPU selection needs no CUDA.
GPU selection never falls back to CPU, MPS or simulation. V0/V1 no longer exist
as independent source files: the repo consolidated them into `pytorch`. Relabeling
that single implementation as three measurements would be misleading.

Defaults preserve the unified sweep workload: combined 70,000 x 784 Fashion-MNIST
float32, pixels divided by 255, shared RandomState(42) initial centers for each K,
K=[2,4,8,16,32,64], exactly 20 Lloyd updates and final assignment/inertia, five runs,
and one CPU library thread. Data distribution, reference and validation tolerances
are unchanged. Download/normalization and centroid selection occur outside timing.
`--iterations`, `--threads`, `--k`, `--runs` explicitly change the experiment and are
recorded. `--smoke` uses 256 rows, preserves all Fashion features, and defaults to
K=2,4; synthetic smoke uses 256 x 4 seeded standard-normal data. It is not a
performance benchmark. Fashion download failures stop execution without fallback.

**Memory:** full-size NumPy broadcasts N x K x D. K=64 can need over 28 GB host RAM
with overlapping temporaries; a T4 Kaggle session may not have enough host memory.
Use the GPU suite there, and a sufficiently large CPU host for the full comparison.
The profiler does not silently shrink data or skip implementations.

## Two timing views; never add them

The implementation code contains optional, sequential scopes. Normal fits execute
these with a no-op context; no CUDA events are created unless profiling is active.
The profiling runner uses the existing complete-fit adapters, kernel dispatches and
validation, rather than a copied algorithm. Both the independent V2.0 host loop and
the V2.1–V2.6 shared loop are instrumented. Normal early stopping remains unchanged;
this runner disables it to preserve fixed work.

1. **Host wall view (`wall_seconds`):** `perf_counter` measures each exclusive scope,
   including launches, any blocking API calls and the profiler work inside that
   scope. `unscoped_host` is elapsed minus all measured scopes. It includes Python
   between scopes, fit returns, cleanup and bookkeeping. The sum of wall scopes
   equals that run's profiled complete-fit elapsed time; this identity is checked.
   It is **not CPU execution time**: blocking host calls can wait on the GPU.
2. **CUDA stream view (`event_seconds`):** timing-enabled start/end events bracket
   device-bearing scopes on PyTorch's current stream or Numba's default stream.
   The existing code uses those streams without concurrent user streams. Events
   are resolved only after the final device fence. There is no added per-stage
   synchronization. These are **stream intervals, not isolated kernel-active
   durations**. An idle device waiting for Python to enqueue a kernel, allocation,
   pageable-memory staging and host gaps between the two records can inflate an
   interval, especially on tiny workloads. They do not measure all device activity
   or capture unrelated/custom-stream work. Use Nsight Systems/Compute for active
   kernel, occupancy, memory throughput, or instruction-level explanations.

Host waits overlap GPU execution. Never add wall and event values or calculate
CPU overhead as elapsed minus event totals. `cpu_overhead` is explicitly N/A:
this profiler cannot isolate pure CPU overhead reliably. CPU preparation and launch
wall costs are visible, with their limits, in the host scopes below. Empty CSV
fields mean N/A; measured zeros remain zeros. Absent stages are not imputed.

## Exact scope definitions

| Stage | Boundary and limitations |
|---|---|
| `initial_work` | Existing input checks, initial centers and fit preparation. Sklearn also copies/centers X and allocates working arrays. PyTorch checks finite tensors with `.item()`, clones centers and creates sample counts; this includes GPU work and implicit waits. Baseline V2.0 retains its existing, lighter validation. |
| `h2d` | Actual input/center transfers, plus Numba inertia-zero upload. Existing `as_tensor`/`to_device` APIs combine allocation and transfer; host wall and event intervals include their setup/staging. **Pure DMA time is N/A.** No fake separation or change to pinned memory. |
| `layout_and_h2d` | V2.2 only: NumPy contiguous transpose plus its extra upload. CPU transpose and DMA cannot be separated by this event interval. Other versions report N/A. |
| `allocation` | Numba's explicit working-device allocations (including shard scratch). No event value; allocation can block. Other backends allocate inside existing operations, so separate allocation is N/A. |
| `assignment` | Each recurring NumPy norm+argmin, PyTorch GEMM scores+argmin, or Numba assignment kernel. Distance and argmin are intentionally grouped. Excludes final assignment. |
| `lloyd_fused` | Each sklearn compiled Lloyd call includes assignment and centroid update/relocation. Separate assignment, accumulation and update are N/A for sklearn. |
| `centroid_reset` | Numba reset of sums, counts and shift, once per update. |
| `centroid_accumulation` | Numba baseline global accumulation; V2.5/V2.6 shard clear + accumulate + merge. The latter three are one explicitly defined scope. |
| `centroid_update` | Numba divide/update kernel. NumPy groups copy, cluster selection and means; PyTorch groups sum/count allocations, index_add, division and empty-center retention. Separate accumulation is N/A for NumPy/PyTorch because those expressions remain together. |
| `final_assignment_inertia` | Final labels and residual inertia after updates. Sklearn also finalizes model fields and restores center offsets. PyTorch `.item()` copies scalar inertia and implicitly waits here, so that scalar D2H cannot be separated. |
| `synchronization` | Host duration of explicit Numba end-of-fit synchronization and runner's final fence; event N/A. Initial boundary fence is outside elapsed. **Not total synchronization cost**: implicit waits in transfers, allocations and `.item()` remain in their caller's scope. |
| `d2h` | Actual host label/center copies, plus Numba inertia copy. Includes allocation/staging and `.numpy()` conversion; pure DMA N/A. PyTorch scalar inertia is already copied in final work. CPU implementations have N/A. |
| `unscoped_host` | Exact host-wall accounting residual, not an estimate of pure CPU overhead. Event N/A. |
| `cpu_overhead` | N/A for every backend; no unsupported subtraction-based estimate. |

`steps.csv` records invocation order and the 1-based Lloyd `step`; preparation,
finalization and fences have an empty step. `stages.csv` sums each stage within a
single fit, never adds a parent scope to children, and includes N/A rows.
Final work is counted once, outside recurring stages. Per-fit cleanup and any
unscoped operations remain in the residual.

## Warmup and observer overhead

Contexts, data loading and independent reference computation are outside timing.
Each configuration/K first runs one full unprofiled warmup (which includes any
first-use JIT/library setup **and execution**), then one full instrumented warmup.
Their elapsed times are in `environment.json`; neither enters statistics. There
is no claim to isolate compiler time. Caches may already be warm from another
configuration or persisted JIT cache; no cold-start measurement is claimed.

Each repeated sample then runs an unprofiled fit and an instrumented fit. Their
order alternates by run; configuration order rotates across K, as in the unified
sweep. Both start with the same host inputs and shared initial centers, allocate
fresh state, and return CPU outputs. A pre-fit fence is excluded and a final fence
included. Every result, including warmups, passes the existing independent float64
reference/objective, nearest-center and inertia checks **outside** elapsed time.

Creating and recording events, reading the clock and managing scopes adds work
and can change launch gaps, allocator behavior, clocks and cache state. The paired
profiled/unprofiled median ratio in `report.md` exposes this perturbation; it does
not remove it or establish causality. The inactive no-op scope calls also add a
small cost compared with older code, especially in synthetic smoke tests. No-op
calls remain in both paired modes. Old results must be rerun for direct comparisons.
Do not use profiled fits to rank close competitors without checking unprofiled
latencies. No claim that V2.3 wins, or that any optimization causes a speedup, is
made until the relevant GPU measurements support it.

## Outputs and interpretation

| File | Contents |
|---|---|
| `runtime.csv` | Validated individual unprofiled/profiled complete-fit seconds, inertia and reference inertia. |
| `steps.csv` | Every exclusive measured invocation, step and ordinal, wall and event seconds. |
| `stages.csv` | Per-fit stage totals and call counts; blank values are N/A. |
| `summary.csv` | Per configuration/K/mode/stage/view: sample count, median, sample SD, quartiles, min/max. A single run has N/A sample SD. |
| `environment.json` | Protocol `stage_profile_v1`, completion/failure, dataset, options, device, installed versions, source revision/dirty state, platform and excluded warmups. |
| `report.md` | Complete-fit comparison and instrumentation latency ratios. |
| `elapsed.png` | Unprofiled/profiled medians with run sample SD. |
| `host_breakdown.png` | Stacked **means** of exclusive wall stages; sums equal mean profiled elapsed. |
| `cuda_intervals.png` | Stacked mean measured device-bearing stream intervals; CPU-only runs show N/A. Does not account for end-to-end latency. |

Do not add per-stage medians: a sum of medians need not equal the median of a sum.
The host breakdown uses means for accounting; summary CSV retains medians and
variability. Neither statistical summary is a confidence interval.
`stage_profile_v1` is separate from `complete_fit_v1` and historical
transfer-excluding results. Keep output directories separate; never combine their
CSV schemas or compare them as if they used identical timing protocols.

Once device and dataset setup succeeds, only the nine known output files are
cleared. Validated results checkpoint after each fit. An exception marks metadata
failed, retains partial raw CSVs and removes final summary/report/plots. A completed
report is written only after every selected configuration passes. Reusing a folder
replaces its profiling artifacts; unrelated files are preserved.

The checked-in [CPU synthetic example](examples/cpu_profile/report.md) demonstrates
real output from this Mac, using the settings recorded in its metadata. These tiny
samples show functionality and instrumentation overhead, not Fashion-MNIST or GPU
performance. No new T4 measurements have been fabricated.

## Validation status

Local tests check real CPU output equivalence with profiling on/off, fixed-step
counts, exact host accounting, event resolution only after the final fence,
non-nesting, N/A serialization, invalid selection and failed-output cleanup.
Existing algorithm tests and non-warp CUDA simulator checks cover preserved
semantics. The real-GPU test profiles every current GPU configuration and checks
results, iteration/event counts and host accounting. Set
`NUMBA_TEST_REQUIRE_GPU=1` so missing NVIDIA CUDA fails instead of skipping it.

CUDA compilation, actual event behavior, warp kernels and full Fashion-MNIST/T4
runs remain unverified on this Apple Silicon Mac. Simulator execution cannot
validate timings and is rejected by the benchmark. On a T4, run the hardware test
above and the existing `test_numba_cuda.py` hardware suite before the full profile.
