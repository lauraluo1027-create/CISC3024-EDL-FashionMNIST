"""Environment setup & verification for the EDL FashionMNIST project.

Ensures torch, torchvision and matplotlib are importable (installs any that
are missing via pip) and verifies that PyTorch's Apple Silicon MPS backend
works end-to-end with a tensor smoke test.

Usage:  python3 setup_env.py
"""

import importlib
import subprocess
import sys

REQUIRED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "matplotlib": "matplotlib",
}


def ensure_packages(packages):
    for module, pip_name in packages.items():
        try:
            importlib.import_module(module)
            print(f"[ok] {module} already installed")
        except ImportError:
            print(f"[..] {pip_name} missing — installing via pip ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"[ok] {pip_name} installed")


def verify_mps():
    import torch

    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        sys.exit(
            "ERROR: PyTorch MPS backend is not available. "
            "This project must run on an Apple Silicon Mac with MPS support."
        )
    x = torch.rand(64, 10, device="mps")
    y = torch.nn.functional.softplus(x).sum()
    torch.mps.synchronize()
    print(f"[ok] MPS tensor smoke test passed (softplus sum = {y.item():.2f})")


if __name__ == "__main__":
    ensure_packages(REQUIRED)
    import matplotlib
    import torch
    import torchvision

    print(f"[ok] torch       {torch.__version__}")
    print(f"[ok] torchvision {torchvision.__version__}")
    print(f"[ok] matplotlib  {matplotlib.__version__}")
    verify_mps()
    print("Environment ready — device to use: 'mps' (Apple Silicon M2)")
