"""RIR-informed late-reverberation suppression.

STFT-domain spectral suppressor following the Lebart (1998) / Habets late
reverberation model: the reverb tail power at time t is predicted from the
signal's own past, decayed at the frequency-dependent rate measured from the
room impulse response (RT60 per octave band). A Wiener-style gain attenuates
time-frequency cells dominated by predicted reverb, floored at the user's
chosen maximum reduction in dB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

from .analysis import RoomProfile


@dataclass
class DereverbSettings:
    reduction_db: float = 6.0  # maximum suppression of reverb-dominated cells
    late_start_ms: float = 60.0  # where "late" reverb begins after direct sound
    window: int = 4096
    hop_divisor: int = 4
    psd_smooth: float = 0.7  # recursive smoothing of the observed PSD (0..1)
    gain_smooth_time: int = 3  # gain smoothing extent in frames
    gain_smooth_freq: int = 5  # gain smoothing extent in bins


def dereverb(
    audio: np.ndarray,
    sr: float,
    profile: RoomProfile,
    settings: DereverbSettings | None = None,
) -> np.ndarray:
    """Suppress late reverberation in ``audio`` (shape: samples or samples x channels).

    Returns processed audio with the same shape and length."""
    settings = settings or DereverbSettings()
    if settings.reduction_db <= 0:
        return audio.copy()

    x = np.atleast_2d(audio.T).astype(np.float64)  # channels x samples
    n_samples = x.shape[1]
    win = settings.window
    hop = win // settings.hop_divisor

    freqs, _, stfts = signal.stft(
        x, fs=sr, window="hann", nperseg=win, noverlap=win - hop, padded=True
    )
    # stfts: channels x bins x frames
    power = np.mean(np.abs(stfts) ** 2, axis=0)  # bins x frames, averaged over channels

    # Recursive smoothing of the observed PSD (reduces estimator variance)
    alpha = settings.psd_smooth
    psd = np.empty_like(power)
    psd[:, 0] = power[:, 0]
    for n in range(1, power.shape[1]):
        psd[:, n] = alpha * psd[:, n - 1] + (1.0 - alpha) * power[:, n]

    # Frequency-dependent decay from the measured RT60:
    # power decays by 60 dB over RT60 -> factor 10^(-6*dt/RT60) over dt seconds
    rt60 = profile.rt60_at(freqs)  # seconds per bin
    frame_dt = hop / sr
    delay_frames = max(1, int(round(settings.late_start_ms / 1000.0 / frame_dt)))
    decay = np.power(10.0, -6.0 * (delay_frames * frame_dt) / np.maximum(rt60, 0.05))

    # Predicted late-reverb PSD: past PSD, delayed and decayed per bin
    late = np.zeros_like(psd)
    late[:, delay_frames:] = psd[:, :-delay_frames] * decay[:, None]

    # Wiener gain with suppression floor
    g_min = 10.0 ** (-settings.reduction_db / 20.0)
    sig = np.maximum(psd - late, 0.0)
    gain = np.where(psd > 0, sig / np.maximum(psd, 1e-30), 1.0)
    gain = np.maximum(gain, g_min)

    # 2D smoothing of the gain matrix against musical noise
    if settings.gain_smooth_time > 1 or settings.gain_smooth_freq > 1:
        gain = uniform_filter(gain, size=(settings.gain_smooth_freq, settings.gain_smooth_time))
        gain = np.clip(gain, g_min, 1.0)

    # Same gain for all channels: preserves the stereo image
    stfts_out = stfts * gain[None, :, :]
    _, y = signal.istft(stfts_out, fs=sr, window="hann", nperseg=win, noverlap=win - hop)
    y = y[:, :n_samples]

    out = y.T if audio.ndim > 1 else y[0]
    return out.reshape(audio.shape).astype(audio.dtype if audio.dtype.kind == "f" else np.float64)
