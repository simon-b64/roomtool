# How roomtool works

This document explains the signal processing behind the two commands, in the
order they are used. It assumes basic familiarity with digital audio, but all
acoustics concepts are introduced along the way.

## Background: what a room does to sound

When a sound is produced in a room, a microphone picks up three things:

1. **Direct sound** — the straight-line path from source to microphone.
2. **Early reflections** — the first bounces off walls, floor and ceiling,
   arriving within roughly the first 50–80 ms. The ear mostly merges these
   with the direct sound; they add a sense of space and size.
3. **Late reverberation** — the dense wash of thousands of overlapping
   reflections that follows. Its level decays exponentially, and the decay
   rate depends on frequency (carpets and audiences absorb treble, so high
   frequencies usually die faster than bass).

All of this is captured by the **room impulse response (RIR)**: the room's
output when excited by an ideal, infinitely short click. Any recording made
in the room is (per source) the dry signal *convolved* with an impulse
response. REW measures the RIR with a swept sine and gives us exactly this
function.

The key number describing the late tail is **RT60**: the time it takes the
reverberant energy to decay by 60 dB. A living room might have RT60 ≈ 0.4 s;
the room this tool was built for measures almost 6 s in the midrange —
cathedral territory.

## Step 1: `roomtool analyze` — measuring the room from its impulse response

### Parsing the REW export

`rew_import.py` reads REW's text export. The header carries the sample
interval (hence sample rate), the peak index (where the direct sound is) and
the start time. The remaining lines are one amplitude value per sample. WAV
exports are accepted too.

### Octave-band splitting

Because decay rates are frequency dependent, the IR is filtered into octave
bands centered at 63, 125, 250, 500, 1000, 2000, 4000 and 8000 Hz (4th-order
Butterworth bandpass, run forward and backward so the filter itself adds no
phase distortion).

### Schroeder backward integration

You cannot read RT60 directly off a raw decay — the IR is a noisy, wiggly
curve. Schroeder (1965) showed that integrating the squared IR *backwards in
time* yields the ensemble-average decay curve in one deterministic step:

```
EDC(t) = ∫t..∞ h²(τ) dτ        (normalized, expressed in dB)
```

This Energy Decay Curve is smooth and ideal for line fitting. One practical
trap: measurement noise makes the EDC flatten into a shelf near the end, which
would bias the fit. The code estimates the noise floor from the last 10 % of
the band-filtered IR and truncates the EDC a safety margin above it (a
simplified version of Lundeby's method).

### Fitting the decay

A straight line is fitted to the EDC between standard evaluation ranges
defined by ISO 3382 (the same standard REW follows, which is why the numbers
are directly comparable):

- **EDT**  (early decay time): 0 to −10 dB
- **T20**: −5 to −25 dB
- **T30**: −5 to −35 dB

The fitted slope is extrapolated to −60 dB to give RT60. T30 is preferred
when the measurement has enough signal-to-noise ratio, T20 otherwise; the
r² of the fit is reported so you can judge reliability.

The analysis also reports broadband **DRR** (direct-to-reverberant energy
ratio) and **clarity** indices C50/C80 (early-vs-late energy within 50/80 ms,
standard metrics for speech and music intelligibility).

The result — RT60 per octave band — is saved as `room_profile.json`. This is
the "fingerprint" of the room that drives step 2.

## Step 2: `roomtool dereverb` — reducing the reverb in a recording

### Why not simply "divide out" the impulse response?

If recording = dry ∗ RIR, inverse filtering looks tempting. It fails here for
three reasons:

1. The RIR is only valid for **one** source and microphone position. Our IR
   was measured at different positions than the recording, and a choir plus
   orchestra is dozens of sources, each with its own RIR.
2. Room responses are non-minimum-phase; exact inverses are unstable or
   non-causal and amplify noise enormously.
3. Even tiny errors (temperature drift, people in the room) break
   sample-exact deconvolution.

So instead of using the IR *exactly*, we use it *statistically*: the one
thing that is position-independent to a good approximation is the **decay
rate** of the diffuse late field — precisely the per-band RT60 we measured.

### The late-reverb model (Lebart 1998 / Habets)

Work in the short-time Fourier transform (STFT) domain: the recording becomes
a grid of time frames × frequency bins (default window 4096 samples, 75 %
overlap, Hann). In this domain a simple but effective model holds:

> The late reverberation power at time t and frequency f is the signal power
> that was present Δ seconds ago, decayed by the room's measured decay rate.

Formally, with the power decaying 60 dB per RT60 seconds:

```
P_late(n, f) = 10^(−6·Δ / RT60(f)) · P_signal(n − Δ, f)
```

where Δ (`--late-start`, default 60 ms) marks where "late" reverb begins —
everything earlier is treated as direct sound plus early reflections, which
are left untouched on purpose (they carry localization and naturalness).
RT60(f) comes from the room profile, interpolated across frequency on a log
axis. The observed power spectrum is smoothed recursively over time before
being used, to keep the estimate stable.

This is where the measurement actually matters: in a hall where bass rings
for 5 s but treble dies in 1.3 s, the predicted reverb — and therefore the
amount of cleaning — automatically follows the right curve per frequency.

### The suppression gain

For every time-frequency cell, estimate how much of the observed power is
reverb, and attenuate accordingly with a Wiener-style gain:

```
G(n, f) = max( (P_obs − P_late) / P_obs ,  G_min )
```

- Cells dominated by predicted reverb get pushed down.
- Cells dominated by direct sound pass through nearly unchanged.
- `G_min = 10^(−reduction_dB / 20)` is the floor: the **maximum** attenuation
  any cell can receive. This is the user's `--reduction` knob. It is a
  ceiling on damage: even where the model is wrong, nothing is ever removed
  by more than the chosen dB amount. With `--reduction 0` the gain is 1
  everywhere and the output is bit-identical to the input (tested).

Raw per-cell gains would flicker, producing the watery "musical noise"
artifact familiar from cheap noise reduction. The gain matrix is therefore
smoothed over neighboring frames and frequency bins before being applied.

### Stereo handling

The gain is computed once, from the channel-averaged power spectrum, and the
**same** gain is applied to both channels. If channels were processed
independently, moment-to-moment gain differences would make instruments
wander across the stereo image. Identical gains change only the level of each
time-frequency cell, never the inter-channel relationship, so the image stays
put.

Finally the modified spectrogram is inverted (overlap-add iSTFT) back to a
waveform of exactly the original length.

### The residual output

`--residual` writes `input − output`: the material that was removed. This is
the fastest honest quality check. Done right, the residual sounds like pure
reverb wash — the room without the music. If you can hear dry instruments or
voices in it, the settings are eating the direct sound: reduce `--reduction`
or increase `--late-start`.

## Why it cannot remove everything

Within the first Δ milliseconds after each note, reverb and direct sound
overlap in the same time-frequency cells — a single microphone cannot
separate them there, and a gain can only attenuate a cell, not un-mix it.
Dense orchestral textures keep *every* cell partially occupied, which is why
sustained tutti passages clean up less than sparse passages and pauses. In
practice 3–10 dB of tail reduction is achievable before artifacts (dulled
attacks, chorus-like warble) outweigh the benefit. This is a fundamental
limit shared by all single-channel spectral methods, not a bug.

## References

- M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust.
  Soc. Am. 37, 1965.
- ISO 3382-1/-2, "Measurement of room acoustic parameters."
- K. Lebart, J.-M. Boucher, P. N. Denbigh, "A New Method Based on Spectral
  Subtraction for Speech Dereverberation," Acta Acustica 87, 2001 (orig. AES
  1998).
- E. A. P. Habets, "Single- and Multi-Microphone Speech Dereverberation using
  Spectral Enhancement," PhD dissertation, TU Eindhoven, 2007.
- A. Lundeby et al., "Uncertainties of Measurements in Room Acoustics,"
  Acustica 81, 1995.
