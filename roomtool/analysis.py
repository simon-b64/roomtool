"""Reverberation analysis of a room impulse response per ISO 3382.

Octave-band RT60 (EDT, T20, T30) via Schroeder backward integration,
plus broadband direct-to-reverberant ratio and clarity (C50/C80).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import signal

from .rew_import import ImpulseResponse

OCTAVE_CENTERS = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


@dataclass
class BandResult:
    center_hz: float
    edt: float | None
    t20: float | None
    t30: float | None
    t20_r2: float | None
    t30_r2: float | None

    @property
    def rt60(self) -> float | None:
        """Best available RT60: T30 preferred, then T20, then EDT."""
        for v in (self.t30, self.t20, self.edt):
            if v is not None:
                return v
        return None


@dataclass
class RoomProfile:
    sample_rate: float
    bands: list[BandResult] = field(default_factory=list)
    drr_db: float | None = None
    c50_db: float | None = None
    c80_db: float | None = None
    source: str = ""

    def rt60_at(self, freqs: np.ndarray) -> np.ndarray:
        """Interpolate RT60 (seconds) at arbitrary frequencies (log-f interpolation)."""
        centers = [b.center_hz for b in self.bands if b.rt60 is not None]
        values = [b.rt60 for b in self.bands if b.rt60 is not None]
        if not centers:
            raise ValueError("No valid RT60 in any band")
        logc = np.log10(centers)
        f = np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)
        return np.interp(np.log10(f), logc, values)

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        for b, bd in zip(self.bands, data["bands"]):
            bd["rt60"] = b.rt60
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RoomProfile":
        data = json.loads(Path(path).read_text())
        bands = [
            BandResult(
                center_hz=b["center_hz"],
                edt=b.get("edt"),
                t20=b.get("t20"),
                t30=b.get("t30"),
                t20_r2=b.get("t20_r2"),
                t30_r2=b.get("t30_r2"),
            )
            for b in data["bands"]
        ]
        return cls(
            sample_rate=data["sample_rate"],
            bands=bands,
            drr_db=data.get("drr_db"),
            c50_db=data.get("c50_db"),
            c80_db=data.get("c80_db"),
            source=data.get("source", ""),
        )


def _octave_filter(ir: np.ndarray, sr: float, center: float) -> np.ndarray | None:
    lo = center / np.sqrt(2.0)
    hi = center * np.sqrt(2.0)
    nyq = sr / 2.0
    if hi >= nyq * 0.95:
        hi = nyq * 0.95
    if lo >= hi:
        return None
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, ir)


def _schroeder_db(band_ir: np.ndarray) -> np.ndarray:
    energy = band_ir.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc = edc / edc[0]
    return 10.0 * np.log10(np.maximum(edc, 1e-30))


def _fit_decay(edc_db: np.ndarray, sr: float, db_start: float, db_end: float) -> tuple[float | None, float | None]:
    """Linear fit of the decay curve between two dB-down points.

    Returns (RT60 extrapolated to -60 dB, r^2 of the fit)."""
    idx_start = np.argmax(edc_db <= db_start)
    idx_end = np.argmax(edc_db <= db_end)
    if idx_end <= idx_start or edc_db[idx_end] > db_end:
        return None, None
    seg = edc_db[idx_start : idx_end + 1]
    t = np.arange(len(seg)) / sr
    slope, intercept = np.polyfit(t, seg, 1)
    if slope >= 0:
        return None, None
    pred = slope * t + intercept
    ss_res = float(np.sum((seg - pred) ** 2))
    ss_tot = float(np.sum((seg - seg.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    rt60 = -60.0 / slope
    return float(rt60), r2


def _truncate_at_noise_floor(edc_db: np.ndarray, band_ir: np.ndarray, sr: float) -> np.ndarray:
    """Crude Lundeby-style truncation: estimate noise floor from the last 10 %
    of the band IR and cut the EDC 5 dB above it to avoid the integration tail
    flattening the fit."""
    n = len(band_ir)
    tail = band_ir[int(n * 0.9) :]
    noise_power = float(np.mean(tail**2))
    total = float(np.sum(band_ir.astype(np.float64) ** 2))
    if noise_power <= 0 or total <= 0:
        return edc_db
    # EDC value when remaining energy equals noise energy over remaining length
    floor_db = 10.0 * np.log10(max(noise_power * n * 0.1 / total, 1e-30)) + 5.0
    cut = np.argmax(edc_db <= floor_db)
    if cut > int(0.05 * sr):
        return edc_db[:cut]
    return edc_db


def analyze(ir: ImpulseResponse, centers: list[float] | None = None) -> RoomProfile:
    centers = centers or OCTAVE_CENTERS
    samples = ir.samples
    peak = ir.peak_index if ir.peak_index is not None else int(np.argmax(np.abs(samples)))

    # Trim leading silence before the direct sound (keep 2.5 ms pre-peak)
    pre = int(0.0025 * ir.sample_rate)
    trimmed = samples[max(0, peak - pre) :]
    peak_t = min(pre, peak)

    profile = RoomProfile(sample_rate=ir.sample_rate, source=ir.source)

    for center in centers:
        band = _octave_filter(trimmed, ir.sample_rate, center)
        if band is None:
            continue
        edc = _schroeder_db(band)
        edc = _truncate_at_noise_floor(edc, band, ir.sample_rate)
        edt, _ = _fit_decay(edc, ir.sample_rate, 0.0, -10.0)
        t20, t20_r2 = _fit_decay(edc, ir.sample_rate, -5.0, -25.0)
        t30, t30_r2 = _fit_decay(edc, ir.sample_rate, -5.0, -35.0)
        profile.bands.append(
            BandResult(center_hz=center, edt=edt, t20=t20, t30=t30, t20_r2=t20_r2, t30_r2=t30_r2)
        )

    # Broadband direct/reverberant split at 2.5 ms after the peak
    split = peak_t + int(0.0025 * ir.sample_rate)
    e = trimmed.astype(np.float64) ** 2
    direct = float(np.sum(e[:split]))
    reverb = float(np.sum(e[split:]))
    if direct > 0 and reverb > 0:
        profile.drr_db = 10.0 * np.log10(direct / reverb)

    for ms, attr in ((50, "c50_db"), (80, "c80_db")):
        k = peak_t + int(ms / 1000.0 * ir.sample_rate)
        early = float(np.sum(e[:k]))
        late = float(np.sum(e[k:]))
        if early > 0 and late > 0:
            setattr(profile, attr, 10.0 * np.log10(early / late))

    return profile


def plot_analysis(ir: ImpulseResponse, profile: RoomProfile, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for band in profile.bands:
        filt = _octave_filter(ir.samples, ir.sample_rate, band.center_hz)
        if filt is None:
            continue
        edc = _schroeder_db(filt)
        t = np.arange(len(edc)) / ir.sample_rate
        step = max(1, len(edc) // 4000)
        ax.plot(t[::step], edc[::step], label=f"{band.center_hz:.0f} Hz", linewidth=0.9)
    ax.set_ylim(-70, 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy decay (dB)")
    ax.set_title("Schroeder decay curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    centers = [b.center_hz for b in profile.bands]
    rt60s = [b.rt60 if b.rt60 is not None else 0.0 for b in profile.bands]
    ax.bar(range(len(centers)), rt60s, color="#4878a8")
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.0f}" for c in centers])
    ax.set_xlabel("Octave band (Hz)")
    ax.set_ylabel("RT60 (s)")
    ax.set_title("RT60 per octave band")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
