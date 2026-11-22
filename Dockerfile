# Numba JIT needs NVVM + libdevice: a CUDA runtime-only image is insufficient.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_HOME=/usr/local/cuda \
    NUMBA_CUDA_USE_NVIDIA_BINDING=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-docker.txt ./
RUN python3 -m pip install --no-cache-dir pip==25.1.1 \
    && python3 -m pip install --no-cache-dir -r requirements-docker.txt \
    && python3 -m pip check \
    && python3 -c "import torch; from numba import cuda; assert torch.version.cuda == '12.4'; assert 'numba_cuda' in cuda.__file__, cuda.__file__; print('CUDA provider:', cuda.__file__)"

COPY implementations/ ./implementations/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/
RUN mkdir -p datasets results

# No GPU is required during build. Pass --gpus all at runtime.
CMD ["python3", "-m", "benchmarks.sweep_k", "--suite", "gpu", "--speedup"]
