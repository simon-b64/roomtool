# roomtool

Analyze a measured room impulse response (from REW) and use it to reduce the
reverberation in recordings made in that room, by an adjustable dB amount.

## Disclaimer

This project was **vibe coded**: written by an AI coding agent (Claude) under
human direction, optimized for "does the output sound good", not for
correctness guarantees. It has not been formally reviewed or validated. The
author takes **no responsibility whatsoever** for anything this software does —
including, but not limited to, wrong analysis results, damaged or lost audio
files, or decisions made based on its output. Use entirely at your own risk,
keep backups of your originals, and verify any results that matter (e.g.
cross-check RT60 values against REW). See [LICENSE](LICENSE) for the formal
no-warranty terms.

## How it works

1. `roomtool analyze` reads the REW impulse response export, splits it into
   octave bands (63 Hz – 8 kHz), and measures the reverberation time per band
   (EDT, T20, T30 via Schroeder backward integration, following ISO 3382 —
   the same method REW uses, so you can cross-check against REW's RT60 display).
   The result is saved as `room_profile.json`, together with a plot of the
   decay curves.
2. `roomtool dereverb` processes a WAV recording in the STFT domain. For each
   frequency, the expected late-reverb energy at any moment is predicted from
   the signal's own recent past, decayed at the rate your room was *measured*
   to decay at (the per-band RT60 from step 1). Time-frequency cells dominated
   by predicted reverb are attenuated with a Wiener-style gain, floored at your
   chosen `--reduction` amount. Stereo files get identical gains on both
   channels, so the stereo image is preserved.

The measured IR is used statistically (decay rates), not sample-exactly —
direct deconvolution would require the IR to be measured at the exact same
source and microphone positions as the recording.

A full explanation of the signal processing, with references, is in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Setting expectations

Removing reverb from choir/orchestra recordings is the hardest case there is.
Realistically the late tail can be reduced by roughly 3–10 dB before artifacts
appear (musical noise, dulled attacks, a "processed" sound). Full removal is
not achievable by any tool. The workflow is: render a sweep of reduction
amounts, listen, and pick the strongest setting that still sounds natural.

## Setup

```
mise install            # provides the pinned Python (3.12)
python -m venv .venv
.venv\Scripts\pip install -e ".[test]"
# optional blind WPE pre-stage:
.venv\Scripts\pip install nara-wpe
```

## Exporting the impulse response from REW

File → Export → Impulse response as text, with these settings:

- **Include the header** — it carries the sample rate, peak index and start
  time, which the parser reads automatically.
- **Do not apply the IR window** (or set the right window to 1000 ms or more):
  the RT60 fit needs the full decay tail.
- **Do not export the minimum-phase version.**
- Normalisation on or off does not matter (all analysis values are ratios).

Alternatively export as WAV (32-bit float, measured IR, no EQ, not minimum
phase) — both formats are accepted.

## Usage

```
# 1. Analyze the room (run once per measurement)
roomtool analyze ir_export.txt
# -> room_profile.json + room_profile.png
# Cross-check the RT60 table against REW's own RT60 display.

# 2. Dereverberate a recording — render several strengths and listen
roomtool dereverb recording.wav --profile room_profile.json --sweep 3 6 10 15 --residual

# 3. Once you know your preferred amount:
roomtool dereverb recording.wav --profile room_profile.json --reduction 6 -o out.wav
```

`--residual` also writes an "input minus output" WAV: listening to what was
removed is the quickest artifact check. If the residual contains dry orchestra
(not just reverb wash), the reduction is set too high.

Tuning flags:

- `--reduction <dB>`: maximum suppression of reverb-dominated time-frequency
  cells. Start at 6.
- `--late-start <ms>` (default 60): where "late" reverb is considered to begin.
  Larger values are gentler and preserve more room character; smaller values
  are more aggressive but risk eating into the direct sound.
- `--window <samples>` (default 4096): STFT window. Smaller (2048) preserves
  transients better; larger (8192) resolves low frequencies better.
- `--wpe`: run a blind WPE (linear prediction) pre-stage before the spectral
  stage. Worth an A/B test; can soften attacks on dense orchestral material.

## Tests

```
.venv\Scripts\python -m pytest tests
```

Covers: RT60 recovery from synthetic IRs with known decay (within 10 %), REW
text/WAV parsing round-trips, `--reduction 0` being a bit-exact no-op, tail
energy actually dropping, and stereo image preservation.
