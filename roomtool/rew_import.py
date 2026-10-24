"""Parse impulse response exports from REW (Room EQ Wizard).

Supported inputs:
- REW "Export impulse response as text" with header: `*`-prefixed comment lines,
  then metadata lines of the form ``<value> // <description>`` (peak value,
  peak index, response length, sample interval, start time), then one sample
  value per line.
- Headerless single-column text (sample rate must be supplied by the caller).
- Two-column text: time (ms or s) and amplitude, optionally with `%` or `*`
  comment lines; sample rate is derived from the time step.
- WAV export of the IR (any format soundfile reads).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImpulseResponse:
    samples: np.ndarray  # float64, mono
    sample_rate: float
    start_time: float = 0.0  # seconds, time of first sample relative to t=0
    peak_index: int | None = None
    source: str = ""

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


_METADATA_KEYS = {
    "peak value": "peak_value",
    "peak index": "peak_index",
    "response length": "response_length",
    "sample interval": "sample_interval",
    "start time": "start_time",
}


def _parse_number(text: str) -> float:
    # REW writes with '.' decimals, but tolerate ',' from locale-formatted files
    return float(text.strip().replace(",", "."))


def load_impulse_response(path: str | Path, sample_rate: float | None = None) -> ImpulseResponse:
    """Load an IR from a REW text export or a WAV file."""
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac", ".aiff", ".aif"):
        return _load_wav(path)
    return _load_text(path, sample_rate)


def _load_wav(path: Path) -> ImpulseResponse:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    peak = int(np.argmax(np.abs(mono)))
    return ImpulseResponse(samples=mono, sample_rate=float(sr), peak_index=peak, source=str(path))


def _load_text(path: Path, sample_rate: float | None) -> ImpulseResponse:
    metadata: dict[str, float] = {}
    values: list[float] = []
    two_col_times: list[float] = []
    two_col = False

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("*") or line.startswith("%"):
                continue
            if "//" in line:
                value_part, _, comment = line.partition("//")
                comment = comment.strip().lower()
                for key, name in _METADATA_KEYS.items():
                    if key in comment:
                        try:
                            metadata[name] = _parse_number(value_part)
                        except ValueError:
                            pass
                        break
                continue
            parts = re.split(r"[\s;,\t]+", line)
            # Guard against ',' used as decimal separator in a single column
            if len(parts) == 2 and "." in line:
                try:
                    t, v = _parse_number(parts[0]), _parse_number(parts[1])
                except ValueError:
                    continue
                two_col = True
                two_col_times.append(t)
                values.append(v)
            else:
                try:
                    values.append(_parse_number(line))
                except ValueError:
                    continue

    if not values:
        raise ValueError(f"No sample data found in {path}")

    samples = np.asarray(values, dtype=np.float64)

    if two_col and len(two_col_times) > 1:
        dt = float(np.median(np.diff(two_col_times)))
        # REW 2-column IR text uses milliseconds; audio-rate steps in seconds
        # would be ~2e-5, so any step above 1e-3 is taken as ms
        scale = 1e-3 if dt > 1e-3 else 1.0
        dt *= scale
        return ImpulseResponse(
            samples=samples,
            sample_rate=1.0 / dt,
            start_time=two_col_times[0] * scale,
            peak_index=int(np.argmax(np.abs(samples))),
            source=str(path),
        )

    if "sample_interval" in metadata and metadata["sample_interval"] > 0:
        sr = 1.0 / metadata["sample_interval"]
    elif sample_rate is not None:
        sr = float(sample_rate)
    else:
        raise ValueError(
            f"{path}: no sample interval in header and no sample rate given. "
            "Re-export from REW with the header included, or pass --sr."
        )

    peak_index = int(metadata["peak_index"]) if "peak_index" in metadata else int(np.argmax(np.abs(samples)))
    start_time = metadata.get("start_time", 0.0)
    return ImpulseResponse(
        samples=samples,
        sample_rate=sr,
        start_time=start_time,
        peak_index=peak_index,
        source=str(path),
    )
