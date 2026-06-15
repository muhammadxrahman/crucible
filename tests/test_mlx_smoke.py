"""M0 acceptance: MLX imports and the Metal GPU is the active device.

Fast and offline. Actual token generation is exercised by scripts/smoke_generate.py,
which downloads a model and is run by hand, not on every push.
"""

import mlx.core as mx


def test_mlx_imports_and_gpu_is_default() -> None:
    assert "gpu" in str(mx.default_device()).lower()


def test_metal_compute_runs_on_gpu() -> None:
    a = mx.ones((256, 256))
    b = (a @ a).sum()
    mx.eval(b)
    assert float(b) == 256 * 256 * 256
