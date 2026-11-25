"""Configure the PyTorch device and thread count for one run."""

import torch


def configure_device(runtime):
    device = runtime["device"]
    torch.set_num_threads(runtime["threads"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device
