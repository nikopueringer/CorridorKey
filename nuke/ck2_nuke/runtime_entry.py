"""Seed external inference without editing the upstream checkout."""

import importlib.util
import os
import random
import runpy
import sys

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(0)
np.random.seed(0)
if importlib.util.find_spec("torch") is not None:
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

script = sys.argv[1]
sys.argv = sys.argv[1:]
# Match direct `python infer.py` module lookup for the separately installed runtime.
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
