# utils

`config.py` -- central `CONFIG` dataclass + `set_seed()` + `DEVICE`. Sets
`CUDA_VISIBLE_DEVICES`/`KERAS_BACKEND` at import time, so import this (or
anything that imports it) before torch/keras/bayesflow.

`plotting.py` -- `show_and_save(fig, name, ...)`, the one function every
XAI/diagnostics plot uses to save to `outputs/figures/` consistently.
