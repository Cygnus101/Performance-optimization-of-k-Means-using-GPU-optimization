# Performance Optimization of K-Means Clustering Using Python-Based GPU Computing (CUDA)

This project investigates how GPU acceleration, mathematical reformulation, and GPU-level optimization affect K-Means computational performance. The target environment is Google Colab with an NVIDIA GPU, using PyTorch with CUDA for the primary GPU implementations and, optionally, Numba for custom CUDA kernels.

K-Means repeatedly assigns samples to their nearest centroid and recomputes centroids. Distance computation makes it a useful course-project workload for studying parallelism, matrix multiplication, and intermediate memory usage. GPU acceleration does not guarantee speedup: workload size, transfer costs, memory requirements, and convergence behavior all matter.

**Current status:** two CPU baselines are implemented and have previously recorded results. The V0 PyTorch CUDA baseline is implemented. GPU benchmark results and the V1 optimization are pending.

## Objectives

- Establish basic and optimized CPU references using NumPy and scikit-learn.
- Compare the baseline PyTorch GPU version (V0) with a matrix multiplication version (V1).
- Measure runtime, speedup, memory consumption, convergence, and clustering correctness.
- Investigate scaling with sample count N, feature dimension D, and cluster count K.
- If time permits, implement and evaluate a custom Numba CUDA assignment kernel (V2).

## Technologies

| Scope | Technologies |
| --- | --- |
| Primary | Python, NumPy, scikit-learn, PyTorch with CUDA, Google Colab, Matplotlib |
| Optional | Numba CUDA, NVIDIA GPU profiling utilities |

The CPU baselines use NumPy and scikit-learn. V0 adds PyTorch with CUDA. Plotting and the optional custom CUDA kernel are planned.

## Implementations and Comparison

| Implementation | Approach | Role |
| --- | --- | --- |
| [Vectorized NumPy K-Means](implementations/vectorized_implementation.py) | CPU implementation using NumPy vectorization | Basic CPU reference |
| [Scikit-learn KMeans](implementations/scikit_implementation.py) | Highly optimized CPU implementation | Strong CPU baseline |
| [V0 PyTorch GPU K-Means](implementations/pytorch_v0_implementation.py) | Standard GPU implementation using `torch.cdist()` with `compute_mode="donot_use_mm_for_euclid_dist"` | Baseline GPU implementation |
| V1: Optimized PyTorch GPU K-Means | Matrix multiplication formulation of squared Euclidean distance | Mathematical and computational optimization |
| V2: Numba CUDA K-Means | Custom GPU kernel for cluster assignment, with explicit control over threads, blocks and memory access | Custom CUDA programming and GPU-level optimization |

V0 and V1 are the primary GPU deliverables. V2 is an optional stretch goal, contingent on available time.

## Mathematical Background

For samples xᵢ and K centroids cⱼ, K-Means minimizes the inertia:

```math
J = \sum_{i=1}^{N} \lVert x_i - c_{a_i} \rVert^2,
\qquad a_i = \arg\min_{j} \lVert x_i - c_j \rVert^2.
```

Each iteration alternates nearest-centroid assignment with computing the mean of each cluster. Empty-cluster behavior must be specified and considered when comparing implementations.

Squared Euclidean distance can be reformulated as:

```math
\lVert x-c \rVert^2 = \lVert x \rVert^2 + \lVert c \rVert^2 - 2x^\top c.
```

For nearest-centroid assignment, the sample norm is constant across centroids, so V1 can use:

```math
a_i = \arg\min_{j} \left(\lVert c_j \rVert^2 - 2x_i^\top c_j\right).
```

For X of shape N × D and C of shape K × D, the cross term is computed as `X @ C.T`. These assignment scores are not themselves squared distances; final inertia must use actual squared residuals or the complete distance expression. Floating-point rounding and ties can produce different assignments, so numerical correctness must be checked.

## Optimization Strategy

### V0: Baseline PyTorch GPU K-Means

- Store samples and centroids as PyTorch tensors on an NVIDIA GPU.
- Compute Euclidean distances using `torch.cdist(X, C, p=2, compute_mode="donot_use_mm_for_euclid_dist")`, explicitly disabling its matrix multiplication path.
- Assign clusters and accumulate centroid sums and counts on the GPU. Keep the previous centroid for any empty cluster, as in the NumPy baseline.
- Use the NumPy baseline's absolute centroid-shift threshold. Each iteration reads the shift back on the host to decide whether to stop.

### V1: Mathematically Optimized PyTorch GPU K-Means — planned

- Use GPU matrix multiplication and centroid squared norms for assignment, omitting the sample norm term.
- Compare execution time against V0 across N, D, and K.
- Investigate intermediate memory usage. Both a straightforward V0 and V1 can materialize N × K arrays; matrix multiplication alone does not eliminate that storage.
- Keep initialization, centroid updates, and stopping rules equivalent to V0 where possible to isolate the assignment change.

V1 is not assumed to outperform V0. Matrix multiplication efficiency, temporary allocations, workload shape, and numerical differences must be evaluated experimentally.

### V2: Custom Numba CUDA K-Means — optional stretch goal

Initially, map one GPU thread to each data point. Each thread computes distances to centroids and selects the closest centroid, avoiding explicit materialization of the complete distance matrix.

If time permits, investigate thread and block configuration, coalesced memory access, shared memory, parallel reductions, and reduced intermediate memory usage. Benchmark each change before drawing conclusions.

Centroid updates may initially remain in PyTorch. This would be a **custom CUDA assignment kernel with PyTorch centroid updates**, not a fully custom CUDA K-Means implementation. Any Numba/PyTorch data exchange and synchronization costs must be accounted for.

## Experimental Setup

**Primary dataset:** Fashion-MNIST, loaded through `sklearn.datasets.fetch_openml`. The existing scripts use the full dataset, convert features to float32, and normalize pixel values by 255. Labels are loaded but are not used to fit K-Means.

**Existing CPU configuration:** seed-42 randomly selected initial centroids, one run, K = 10, at most 1,000 iterations, and tolerance value `1e-4`. Both scripts use the same initial centroids. Their reported clustering times include centroid initialization and fitting, with dataset loading outside the timed section.

**Planned scaling experiments:** synthetic datasets with fixed seeds, varying N, D, and K independently while holding the other dimensions fixed. Choose feasible sizes for the available Colab GPU and report actual dataset shapes and generation settings. No synthetic-data benchmark runner is currently included.

For every measured environment, record the CPU, thread settings, RAM, GPU model and device memory, operating system, Python and dependency versions, CUDA environment, precision, and matrix multiplication settings. Colab hardware availability can vary; report the GPU actually used. The hardware and software environment for the existing recorded CPU results is not documented.

## Benchmark Methodology and Metrics

| Metric | Evaluation |
| --- | --- |
| Execution time | Repeat runs and report a central estimate and variability; identify clustering-only and end-to-end timing scopes. |
| Speedup | Divide each CPU baseline's runtime by the evaluated implementation's runtime for the same workload and timing scope. |
| GPU memory consumption | Report peak memory, measurement method, and whether inputs, intermediates, and reserved memory are included. For V2, account for allocations outside PyTorch's allocator. |
| Iterations and convergence | Report iteration counts, stopping rules, and whether the iteration limit was reached. |
| Final inertia and correctness | Compare squared-residual inertia, valid labels, finite centroids, and nearest-centroid assignments within stated numerical tolerances; account for arbitrary cluster-label ordering. |
| Scaling | Vary N, D, and K independently and plot runtime, speedup, and memory usage. |

Benchmark protocol:

1. Validate numerical correctness before comparing performance. Use small controlled cases as well as Fashion-MNIST; optionally report an external clustering-quality metric where labels are available.
2. Use identical preprocessing, equivalent initialization strategies, fixed random seeds, and the same initial centroids where supported. Match input precision, restart count, maximum iterations, and convergence criteria where possible; document unavoidable differences and empty-cluster handling.
3. Warm up GPU execution. State whether compilation and warm-up costs are excluded or reported separately.
4. Separate host-to-device transfer overhead from computation time, or explicitly report end-to-end time. Include final assignment and inertia computation consistently, or report them separately.
5. Synchronize GPU execution before starting and after finishing timed GPU work so asynchronous execution does not understate runtime.
6. Repeat configurations, document timing and memory measurement methods, and record the hardware/software environment.
7. Interpret runtime together with iterations, convergence, and inertia. Different convergence trajectories or implementation details can change elapsed time without demonstrating a faster equivalent iteration.

**Existing baseline caveats:** the NumPy implementation uses an absolute centroid-shift threshold, whereas scikit-learn uses a relative tolerance. Setting both to `1e-4` does not make their stopping criteria identical. Scikit-learn also handles empty clusters differently. Its reported iteration-limit flag does not establish whether convergence occurred on the last iteration. These differences must remain visible in subsequent comparisons.

## Implementation Status

| Work | Status | Evidence / remaining work |
| --- | --- | --- |
| NumPy CPU baseline | Implemented; recorded results available | [Source](implementations/vectorized_implementation.py); reports runtime, iterations, convergence, and inertia |
| Scikit-learn CPU baseline | Implemented; recorded results available | [Source](implementations/scikit_implementation.py); reports runtime, iterations, iteration-limit flag, and inertia |
| V0 PyTorch GPU | Implemented | [Source](implementations/pytorch_v0_implementation.py). Reports time, peak allocated GPU tensor memory, iterations, convergence, and inertia. CUDA benchmark results are pending. |
| V1 optimized PyTorch GPU | Planned | No implementation present |
| V2 Numba CUDA assignment | Optional stretch goal | No implementation present |
| Repeated/scaling benchmarks, memory measurements, and plots | Planned | No automated experiment runner or plots present |

Systematic correctness checks and repeated CPU and GPU benchmarks are pending.

## Roadmap

1. **Phase 1: CPU baselines.** NumPy and scikit-learn implementations are available. Complete systematic correctness checks and repeated runtime benchmarks, recording the environment and convergence differences.
2. **Phase 2: Baseline GPU acceleration.** Run the V0 PyTorch baseline on CUDA and compare CPU and GPU performance.
3. **Phase 3: Mathematical optimization.** Implement V1 using the matrix multiplication formulation; investigate runtime and memory efficiency.
4. **Phase 4: Custom CUDA programming (optional).** Implement V2 Numba CUDA cluster assignment, followed by kernel-level benchmarking and selected optimizations.
5. **Phase 5: Experimental evaluation.** Benchmark all completed implementations, analyze performance scaling, plot runtime/speedup/memory usage, and summarize findings and limitations.

## Results Currently Available

The CPU measurements below are preserved from the previous README. They have not been rerun for this documentation update. They are individual recorded results, not a repeated-run study; environment details and runtime variability are unavailable. GPU results, speedups, and memory measurements remain pending.

| Implementation | Execution Time | Speedup vs. Vectorized NumPy | Speedup vs. Scikit-learn | Peak Memory | Iterations | Final Inertia |
| --- | --- | --- | --- | --- | --- | --- |
| Vectorized NumPy K-Means | 17.535 seconds | — | — | — | 34 | 2223801.093308 |
| Scikit-learn KMeans | 0.561 seconds | — | — | — | 28 | 2223801.500000 |
| V0: PyTorch GPU K-Means | Pending | — | — | — | — | — |
| V1: Optimized PyTorch GPU K-Means | Pending | — | — | — | — | — |
| V2: Numba CUDA K-Means (optional) | Pending | — | — | — | — | — |

Future result tables should identify N, D, K, precision, initialization, hardware, timing scope, and repeated-run statistics. No conclusion about GPU speedup or the relative performance of V0, V1, and V2 is supported yet.

## Repository Structure

```text
README.md
implementations/
    vectorized_implementation.py   # NumPy CPU baseline and shared dataset loader
    scikit_implementation.py       # scikit-learn CPU baseline
    pytorch_v0_implementation.py   # PyTorch CUDA baseline
```

The dataset loader stores its OpenML cache in `datasets/` at the repository root.

## Setup and Execution

From the repository root, create a Python virtual environment, install the CPU dependencies, and run both baselines. The activation command below is for macOS and Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scikit-learn
python implementations/vectorized_implementation.py
python implementations/scikit_implementation.py
```

The first dataset fetch requires internet access; subsequent runs can reuse `datasets/`. These scripts execute the CPU configuration described above and print clustering time, iteration information, and final inertia. The NumPy implementation constructs a broadcast N × K × D intermediate, so the full Fashion-MNIST run can require substantial host memory.

### Run V0 on an NVIDIA GPU

Use a Google Colab GPU runtime or a Python environment with a [CUDA-enabled PyTorch installation](https://pytorch.org/get-started/locally/). From the repository root, run these commands (prefix each line with `!` if using a Colab code cell).

```bash
python -m pip install numpy scikit-learn
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python implementations/pytorch_v0_implementation.py
```

V0 uses the CPU baselines' Fashion-MNIST preprocessing, seed, cluster count, and iteration cap. The script synchronizes CUDA around the timed run and includes initialization, fitting, final assignment, and inertia. It excludes dataset loading, input transfer, and a warm-up on the first 1,024 samples. Peak allocated PyTorch GPU tensor memory includes the resident input and temporary tensors. Allocator reservations and CUDA allocations outside PyTorch are outside this measurement.

## Future Scope

After V0/V1 correctness and evaluation are complete, consider the optional V2 assignment kernel, memory-saving batching for larger workloads, and targeted GPU profiling. A fully custom CUDA centroid-update stage is a possible later extension, beyond the initial assignment-only stretch goal. Prioritize reproducible comparisons and clear limitations over adding optimization versions without evidence.
