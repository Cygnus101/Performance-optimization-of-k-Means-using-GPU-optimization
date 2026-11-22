"""Independent Numba configurations for the unified sweep and hardware tests."""
import importlib
from implementations.numba_cuda.common import load_baseline


def configurations(thread_sizes):
    configs = [("V2.0 baseline", load_baseline().kmeans, {})]
    tuning = importlib.import_module("implementations.numba_cuda.v2_1_block_tuning")
    for threads in thread_sizes:
        configs.append((f"V2.1 threads={threads}", tuning.kmeans,
                        {"threads_per_block": threads}))
    for module in ("v2_2_coalescing", "v2_3_warp", "v2_4_shared_memory", "v2_5_reduction"):
        fit = importlib.import_module(f"implementations.numba_cuda.{module}").kmeans
        configs.append((module, fit, {}))
    return configs


