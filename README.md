# Performance Optimization of K-Means Clustering Using GPU-Based Parallelization on Apple Silicon

This project investigates how GPU-based parallelization and progressive optimization affect the performance of K-Means clustering on Apple Silicon. The main contribution is a custom MLX/Metal implementation, developed and evaluated through a sequence of optimization versions.

The evaluation has two distinct parts:

1. **Final benchmark comparison:** compare the final custom implementation against three reference implementations.
2. **Sequential optimization and ablation study:** measure how individual changes affect the custom implementation from V0 through V8.

This README defines the evaluation plan. Implementations, optimization details, and measured results will be added as the project progresses; no performance claims are made yet.

Dataset: Fashion-MNIST

## Implementations

| Implementation | Approach | Role in the Project |
| --- | --- | --- |
| Vectorized K-Means | Simple implementation using numpy vectorization. | Standard reference for establishing baseline performance. |
| Scikit-learn KMeans | Optimized CPU-based implementation. | Strong conventional CPU baseline. |
| My Optimized MLX/Metal K-Means | Custom implementation developed progressively through GPU-oriented optimizations on Apple Silicon. | Main implementation and contribution of the project. |
| Flash-KMeans-MLX | Existing highly optimized MLX-based K-Means implementation. | External optimized GPU reference. |

## Final Benchmark Comparison

This comparison evaluates all four implementations on the same workloads. The custom implementation is represented by its final selected version; intermediate versions are evaluated separately in the optimization study below.

### Evaluation Metrics

| Metric | Evaluation |
| --- | --- |
| Execution time | Measure elapsed time over repeated runs, reporting a central estimate and variability. Distinguish end-to-end time from clustering-only time. |
| Speedup | Divide a reference implementation's runtime by the evaluated implementation's runtime for the same workload. Report speedup relative to both Naive K-Means and Scikit-learn KMeans. |
| Memory usage | Measure peak memory usage and state the measurement method and scope. Account for Apple Silicon's unified memory without double-counting shared allocations. |
| Scaling with samples N | Vary the number of samples while holding D and K fixed. |
| Scaling with dimensionality D | Vary the number of features while holding N and K fixed. |
| Scaling with clusters K | Vary the number of clusters while holding N and D fixed. |
| Correctness and convergence | Check valid assignments and centroids, iteration counts, and stopping behavior. Compare results within a stated numerical tolerance while accounting for arbitrary cluster-label ordering. |
| Final inertia or clustering quality | Report the sum of squared distances from samples to their assigned centroids. Where ground-truth labels are available, optionally report an external clustering-quality metric. |

### Benchmark Protocol

- Use identical datasets and preprocessing across implementations. Record dataset characteristics, N, D, K, and random seeds.
- Use the same initial centroids where supported. Match initialization strategy, number of restarts, maximum iterations, convergence tolerance, and numerical precision where possible; document differences.
- Record the Apple Silicon chip, available memory, operating system, dependency versions, CPU thread settings, and the exact Flash-KMeans-MLX source and revision.
- Perform warm-up runs and report whether compilation, data conversion, and device transfer costs are included. Force evaluation and synchronize GPU work before stopping timers so measurements include completed computation.
- Repeat each configuration and report runtime variability. Specify whether memory measurements include input storage and temporary allocations.
- Evaluate performance together with convergence and inertia. Report iteration counts so early stopping or reduced clustering quality is not mistaken for an implementation speedup.

### Results

Results are pending. Populate a comparison table for each workload configuration, identifying N, D, K, precision, and initialization settings. Summarize scaling across configurations with plots or additional tables.

| Implementation | Execution Time | Speedup vs. Naive | Speedup vs. Scikit-learn | Peak Memory | Iterations / Convergence | Final Inertia / Quality |
| --- | --- | --- | --- | --- | --- | --- |
| Naive K-Means | — | — | — | — | — | — |
| Scikit-learn KMeans | — | — | — | — | — | — |
| My Optimized MLX/Metal K-Means | — | — | — | — | — | — |
| Flash-KMeans-MLX | — | — | — | — | — | — |

## Optimization Progression of My MLX/Metal Implementation

This study tracks only the custom implementation. V0 establishes its starting point, and subsequent versions record incremental changes. The optimization entries are intentionally blank until the changes are implemented and measured.

| Version | Optimization | Description / Change | Metrics to Evaluate |
| ------- | ------------ | -------------------- | ------------------- |
| V0      |              |                      |                     |
| V1      |              |                      |                     |
| V2      |              |                      |                     |
| V3      |              |                      |                     |
| V4      |              |                      |                     |
| V5      |              |                      |                     |
| V6      |              |                      |                     |
| V7      |              |                      |                     |
| V8      |              |                      |                     |

Evaluate each version using the same workloads and benchmark protocol. Report incremental speedup relative to the preceding version and cumulative speedup relative to V0, alongside memory usage, convergence, and final inertia.

Where practical, introduce one change per version. For versions containing multiple changes, use controlled ablations that disable or isolate individual optimizations to determine their contribution. A cumulative version comparison alone does not establish each optimization's independent effect.

The final benchmark comparison answers **how the completed custom implementation compares with the three references**. The optimization study answers **how and why the custom implementation's performance changes across versions**.
