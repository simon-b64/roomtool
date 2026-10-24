"""Optional blind dereverberation pre-stage using nara-wpe.

Installed via the ``wpe`` extra: ``pip install roomtool[wpe]`` or
``pip install nara-wpe``.
"""

from __future__ import annotations

import numpy as np


def wpe_dereverb(audio: np.ndarray, sr: float, taps: int = 10, delay: int = 3) -> np.ndarray:
    """Run offline WPE on audio (samples or samples x channels)."""
    try:
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import stft, istft
    except ImportError as exc:
        raise RuntimeError(
            "nara-wpe is not installed. Install it with: pip install nara-wpe"
        ) from exc

    x = np.atleast_2d(audio.T)  # channels x samples
    size, shift = 512, 128
    Y = stft(x, size=size, shift=shift)  # channels x frames x bins
    Y = Y.transpose(2, 0, 1)  # bins x channels x frames
    Z = wpe(Y, taps=taps, delay=delay, iterations=3, statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=size, shift=shift)
    z = z[:, : x.shape[1]]
    out = z.T if audio.ndim > 1 else z[0]
    return np.asarray(out, dtype=np.float64).reshape(audio.shape)
