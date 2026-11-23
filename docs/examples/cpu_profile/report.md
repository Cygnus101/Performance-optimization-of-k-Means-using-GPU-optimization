# Comparative stage profile

Measured complete-fit medians; profiled/unprofiled is a latency ratio, not CPU overhead.

| Implementation | K | Unprofiled (s) | Profiled (s) | Ratio |
|---|---:|---:|---:|---:|
| NumPy CPU | 2 | 0.000508 | 0.000520 | 1.023 |
| scikit-learn CPU | 2 | 0.000239 | 0.000248 | 1.041 |
| scikit-learn CPU | 4 | 0.000215 | 0.000254 | 1.183 |
| NumPy CPU | 4 | 0.000836 | 0.000848 | 1.014 |

Empty CSV numeric fields mean N/A, never zero. See docs/profiling.md for stage boundaries, fused operations, event limitations and accounting.
Host breakdown uses means so stacked scopes account for mean elapsed time. Do not sum per-stage medians or add wall and event views.
One unprofiled and one instrumented full warmup per configuration/K are excluded from repeated samples; warmup includes execution and any JIT, not isolated compiler time.
