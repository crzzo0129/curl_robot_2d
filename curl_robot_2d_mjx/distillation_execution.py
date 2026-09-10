"""Execution helpers; batches keep their global leading environment dimension."""

from contextlib import contextmanager
import threading
import time


class BatchExecution:
    def __init__(self, num_devices=1):
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        devices = jax.local_devices()
        if num_devices > len(devices):
            raise ValueError(f"requested {num_devices} devices, only {devices} visible")
        self.jax = jax
        self.devices = devices[:num_devices]
        mesh = Mesh(np.asarray(self.devices), ("env",))
        self.batch = NamedSharding(mesh, PartitionSpec("env"))
        self.replicated = NamedSharding(mesh, PartitionSpec())
        self.description = ", ".join(str(device) for device in self.devices)

    def batch_jit(self, function):
        return self.jax.jit(function, out_shardings=self.batch)

    def train_jit(self, function):
        # Global means in the loss produce cross-device gradient reductions.
        # Replicated outputs prevent independent per-device optimizer updates.
        return self.jax.jit(
            function,
            in_shardings=(self.replicated, self.replicated,
                          self.batch, self.batch, self.batch),
            out_shardings=(self.replicated,) * 4,
        )


@contextmanager
def timed_stage(label, heartbeat_seconds=30):
    """Report progress even while the host is compiling or awaiting a device."""
    started = time.perf_counter()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            print(f"[{label}] running {time.perf_counter() - started:.1f}s "
                  "(compilation/device execution)", flush=True)

    print(f"[{label}] starting", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        print(f"[{label}] finished after {time.perf_counter() - started:.1f}s", flush=True)
