"""Local random-number management that restores global Torch state."""

from __future__ import annotations

from contextlib import contextmanager
import os

import numpy as np
import torch
from sklearn.utils import check_random_state


class SeedManager:
    """Create reproducible local generators from a scikit-learn random state."""

    def __init__(self, random_state=None):
        rng = check_random_state(random_state)
        self.seed = int(rng.randint(0, np.iinfo(np.int32).max))
        self.numpy = np.random.default_rng(self.seed)
        self.torch = torch.Generator(device="cpu").manual_seed(self.seed)

    def seed_worker(self, worker_id: int) -> None:
        """Seed NumPy inside a DataLoader worker from the run-local seed."""
        del worker_id
        np.random.seed(torch.initial_seed() % 2**32)

    def __getstate__(self):
        return {"seed": self.seed, "numpy_state": self.numpy.bit_generator.state}

    def __setstate__(self, state):
        self.seed = state["seed"]
        self.numpy = np.random.default_rng()
        self.numpy.bit_generator.state = state["numpy_state"]
        self.torch = torch.Generator(device="cpu").manual_seed(self.seed)

    @contextmanager
    def torch_fork(self, device: torch.device, *, deterministic: bool = False):
        devices = []
        if device.type == "cuda":
            devices = [device.index or 0]
        previous_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if deterministic and device.type == "cuda":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        previous_algorithms = torch.are_deterministic_algorithms_enabled()
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            if deterministic:
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            try:
                yield
            finally:
                torch.use_deterministic_algorithms(previous_algorithms)
                torch.backends.cudnn.deterministic = previous_cudnn_deterministic
                torch.backends.cudnn.benchmark = previous_cudnn_benchmark
                if deterministic and device.type == "cuda":
                    if previous_cublas_config is None:
                        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                    else:
                        os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas_config
