# Agent context for roomtool

## What this project is

A Python CLI (`roomtool`) that analyzes a measured room impulse response
(exported from REW / Room EQ Wizard) and uses that analysis to reduce
reverberation in WAV recordings made in the same room by a user-chosen dB
amount. Built for one concrete use case: choir + orchestra recordings made in
a very reverberant church-scale room (measured RT60 ≈ 5.9 s at 500 Hz).

**Success is judged by ear.** The user listens to the processed output and the
residual; there is no objective target metric. Expect iteration on tuning
parameters, not on architecture.

## Key design decisions (do not re-litigate without reason)

- **No direct deconvolution.** The IR was measured at different mic/source
  positions than the recordings, and the sources (choir/orchestra) are
  spatially distributed. The IR is used *statistically*: per-octave-band RT60
  drives a late-reverberation spectral suppressor (Lebart 1998 / Habets model).
- **`--reduction <dB>` is the user's main knob** — it is the floor of the
  Wiener gain, i.e. the maximum suppression of reverb-dominated
  time-frequency cells. `--reduction 0` must remain a bit-exact no-op
  (tested).
- **Stereo gets identical gain on both channels** (computed from the averaged
  channel power) to preserve the stereo image. Keep it that way.
- **Dereverberating orchestral music is inherently limited**: ~3–10 dB of tail
  reduction before artifacts (musical noise, dulled attacks). Full removal is
  impossible; do not promise it.
- Sample rates may differ between IR (48 kHz) and recording (96 kHz). That is
  fine by design: the profile stores RT60 vs. frequency and interpolates
  (log-frequency, clamped at the outermost bands).

## Layout

- `roomtool/rew_import.py` — parses REW IR exports: txt with `//`-commented
  header (sample interval, peak index, start time), headerless single-column
  (needs `--sr`), 2-column time/amplitude (ms heuristic), or WAV. Real REW
  files may be Windows-1252 encoded; the parser reads with
  `errors="replace"`, bad bytes only occur in comment lines.
- `roomtool/analysis.py` — octave-band RT60 (EDT/T20/T30) via Schroeder
  backward integration per ISO 3382, with crude Lundeby-style noise-floor
  truncation; DRR, C50, C80. Produces `RoomProfile` (JSON-serializable) and a
  matplotlib plot. REW computes RT60 the same way, so REW's RT60 display is
  the reference for validating changes here.
- `roomtool/dereverb.py` — STFT suppressor. Late-reverb PSD predicted from
  the signal's own past, delayed by `late_start_ms` and decayed by
  `10^(-6*dt/RT60(f))`; Wiener gain floored at `10^(-reduction/20)`; 2D gain
  smoothing against musical noise.
- `roomtool/wpe_stage.py` — optional blind nara-wpe pre-stage (lazy import,
  `pip install nara-wpe`).
- `roomtool/cli.py` — `roomtool analyze` and `roomtool dereverb` (supports
  `--sweep 3 6 10`, `--residual` for input-minus-output).
- `tests/test_roomtool.py` — synthetic IRs with known RT60 (recovery within
  10 %), parser round-trips, dereverb invariants (no-op at 0 dB, tail energy
  drops, stereo ratio preserved, length preserved).

## Environment / workflow

- Windows. Python pinned via mise (`mise.toml`, Python 3.12); packages in
  `.venv` (`.venv\Scripts\python.exe`). Do not write to the user's shared
  `C:\Users\sbuchinger\Developer\mise.toml` — use the project-local one.
- Install: `.venv\Scripts\pip install -e ".[test]"`.
- Run tests before committing: `.venv\Scripts\python -m pytest tests -q`.
- Real input data sits in the repo root: `IR Reihe 6 Mitte.txt` (REW export,
  48 kHz, ~11 s, committed) and `CKK202610.wav` (96 kHz stereo recording,
  NOT committed). All `*.wav` plus `room_profile.*` are gitignored — never
  commit audio files; the recording was deliberately purged from history
  before publishing (performer rights, repo size).
- Keep the README's REW export instructions in sync with parser behavior
  (header required unless `--sr` given; no IR window; not minimum phase).

## Tuning knobs the user iterates on

`--reduction` (suppression floor), `--late-start` (late reverb onset, default
60 ms; lower = more aggressive), `--window` (STFT size, default 4096; consider
8192 for 96 kHz material), `--wpe` (blind pre-stage A/B). Deeper tuning lives
in `DereverbSettings` (PSD smoothing, gain smoothing extents).
