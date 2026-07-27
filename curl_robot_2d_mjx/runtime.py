"""Cloud runtime helpers that stay importable without JAX installed."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
import sys


def select_mujoco_gl_backend(
    environ: dict[str, str] | None = None,
    platform_name: str | None = None,
) -> str:
    """Choose a headless-safe MuJoCo renderer like disk_robot does."""

    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    configured = environ.get("MUJOCO_GL")
    if configured:
        return configured
    if platform_name.startswith("linux") and not environ.get("DISPLAY"):
        return "egl"
    return "glfw"


def configure_cloud_runtime(
    *,
    memory_fraction: float = 0.90,
    preallocate: bool = True,
    xla_triton: bool = True,
    mujoco_gl: str = "auto",
    matmul_precision: str = "high",
    verbose: bool = False,
) -> None:
    """Set GPU defaults before importing JAX.

    CUDA libraries are expected to come from ``jax[cuda12]``.  The setting is
    valid for both RTX 4090 and H200 Linux instances.
    """

    flags = os.environ.get("XLA_FLAGS", "")
    xla_flags: list[str] = []
    if xla_triton:
        xla_flags.extend(
            [
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                "--xla_gpu_shard_autotuning=false",
                "--xla_gpu_triton_gemm_any=True",
            ]
        )
        try:
            jaxlib_version = importlib.metadata.version("jaxlib").replace(
                ".", "_"
            )
        except importlib.metadata.PackageNotFoundError:
            jaxlib_version = None
        if jaxlib_version is not None:
            autotune_path = (
                f"/tmp/xla_autotune_jaxlib_{jaxlib_version}.pbtxt"
            )
            xla_flags.append(
                f"--xla_gpu_dump_autotune_results_to={autotune_path}"
            )
            if Path(autotune_path).exists():
                xla_flags.append(
                    f"--xla_gpu_load_autotune_results_from={autotune_path}"
                )
    for flag in xla_flags:
        if flag not in flags:
            flags = f"{flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = flags
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", f"{memory_fraction:.2f}"
    )
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if preallocate else "false",
    )
    if mujoco_gl == "auto":
        os.environ["MUJOCO_GL"] = select_mujoco_gl_backend()
    elif mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
    if os.environ.get("MUJOCO_GL") not in (None, "", "disable"):
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    if matmul_precision:
        os.environ.setdefault(
            "JAX_DEFAULT_MATMUL_PRECISION", matmul_precision
        )
    cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    if verbose:
        print(
            "stage=runtime_config "
            f"mujoco_gl={os.environ.get('MUJOCO_GL', '')} "
            f"matmul_precision="
            f"{os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '')} "
            f"preallocate="
            f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '')} "
            f"memory_fraction="
            f"{os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '')}",
            flush=True,
        )


def describe_runtime() -> dict[str, object]:
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX is unavailable. Install requirements-mjx.txt on a Linux "
            "NVIDIA instance before running MJX."
        ) from exc
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "compilation_cache": os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", ""
        ),
        "memory_fraction": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION", ""
        ),
    }
