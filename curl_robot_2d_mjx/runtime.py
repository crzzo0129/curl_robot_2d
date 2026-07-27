"""Cloud runtime helpers that stay importable without JAX installed."""

from __future__ import annotations

import os
from pathlib import Path


def configure_cloud_runtime(
    *,
    memory_fraction: float = 0.90,
    preallocate: bool = True,
) -> None:
    """Set GPU defaults before importing JAX.

    CUDA libraries are expected to come from ``jax[cuda12]``.  The setting is
    valid for both RTX 4090 and H200 Linux instances.
    """

    flags = os.environ.get("XLA_FLAGS", "")
    triton_flag = "--xla_gpu_triton_gemm_any=true"
    if triton_flag not in flags:
        flags = f"{flags} {triton_flag}".strip()
    os.environ["XLA_FLAGS"] = flags
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", f"{memory_fraction:.2f}"
    )
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if preallocate else "false",
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")
    cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))


def describe_runtime() -> dict[str, object]:
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX is unavailable. Install requirements-mjx.txt on a Linux "
            "NVIDIA instance before running MJX."
        ) from exc
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "memory_fraction": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION", ""
        ),
    }
