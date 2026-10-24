import io
from pathlib import Path

import numpy as np
import pytest

from roomtool.rew_import import load_impulse_response, ImpulseResponse
from roomtool.analysis import analyze, RoomProfile, BandResult
from roomtool.dereverb import dereverb, DereverbSettings

SR = 48000


def synthetic_ir(rt60: float, seconds: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Exponentially decaying noise with a known broadband RT60, plus a direct spike."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    envelope = 10.0 ** (-3.0 * t / rt60)  # -60 dB at t = rt60
    tail = rng.standard_normal(n) * envelope * 0.1
    ir = tail.copy()
    ir[0] = 1.0  # direct sound
    return ir


class TestAnalysis:
    @pytest.mark.parametrize("rt60", [0.5, 1.2, 2.5])
    def test_recovers_known_rt60(self, rt60):
        ir = ImpulseResponse(samples=synthetic_ir(rt60, seconds=max(2.0, rt60 * 2)),
                             sample_rate=SR, peak_index=0)
        profile = analyze(ir)
        # mid bands are most reliable for broadband noise decay
        mid = [b.rt60 for b in profile.bands if b.center_hz in (500, 1000, 2000) and b.rt60]
        assert mid, "no RT60 recovered in mid bands"
        for value in mid:
            assert value == pytest.approx(rt60, rel=0.10)

    def test_profile_roundtrip(self, tmp_path):
        ir = ImpulseResponse(samples=synthetic_ir(1.0), sample_rate=SR, peak_index=0)
        profile = analyze(ir)
        p = tmp_path / "profile.json"
        profile.save(p)
        loaded = RoomProfile.load(p)
        assert loaded.sample_rate == profile.sample_rate
        assert [b.rt60 for b in loaded.bands] == pytest.approx(
            [b.rt60 for b in profile.bands], rel=1e-9)

    def test_rt60_interpolation(self):
        profile = RoomProfile(sample_rate=SR, bands=[
            BandResult(125, None, None, 2.0, None, 0.99),
            BandResult(1000, None, None, 1.0, None, 0.99),
        ])
        vals = profile.rt60_at(np.array([125.0, 1000.0, 20000.0]))
        assert vals[0] == pytest.approx(2.0)
        assert vals[1] == pytest.approx(1.0)
        assert vals[2] == pytest.approx(1.0)  # clamped above the highest band


class TestRewImport:
    def test_header_text(self, tmp_path):
        ir = synthetic_ir(0.8, seconds=0.5)
        f = tmp_path / "ir.txt"
        lines = [
            "* Impulse Response data saved by REW V5.31",
            "* IR is not normalised",
            "* IR window has not been applied",
            "* IR is not the minimum phase version",
            "1.0 // Peak value before normalisation",
            "0 // Peak index",
            f"{len(ir)} // Response length",
            f"{1.0 / SR:.17E} // Sample interval (seconds)",
            "0.0 // Start time (seconds)",
        ] + [f"{v:.10E}" for v in ir]
        f.write_text("\n".join(lines))
        loaded = load_impulse_response(f)
        assert loaded.sample_rate == pytest.approx(SR)
        assert loaded.peak_index == 0
        assert len(loaded.samples) == len(ir)
        assert loaded.samples == pytest.approx(ir, abs=1e-9)

    def test_headerless_needs_sr(self, tmp_path):
        f = tmp_path / "ir.txt"
        f.write_text("\n".join(f"{v:.8f}" for v in synthetic_ir(0.5, seconds=0.2)))
        with pytest.raises(ValueError, match="sample"):
            load_impulse_response(f)
        loaded = load_impulse_response(f, sample_rate=SR)
        assert loaded.sample_rate == SR

    def test_two_column_ms(self, tmp_path):
        ir = synthetic_ir(0.5, seconds=0.2)
        times_ms = np.arange(len(ir)) / SR * 1000.0
        f = tmp_path / "ir.txt"
        f.write_text("% time amplitude\n" + "\n".join(
            f"{t:.6f} {v:.8f}" for t, v in zip(times_ms, ir)))
        loaded = load_impulse_response(f)
        assert loaded.sample_rate == pytest.approx(SR, rel=1e-3)

    def test_wav(self, tmp_path):
        import soundfile as sf
        ir = synthetic_ir(0.5, seconds=0.5)
        f = tmp_path / "ir.wav"
        sf.write(f, ir / np.max(np.abs(ir)), SR, subtype="FLOAT")
        loaded = load_impulse_response(f)
        assert loaded.sample_rate == SR
        assert loaded.peak_index == 0


def make_profile(rt60: float = 1.5) -> RoomProfile:
    return RoomProfile(sample_rate=SR, bands=[
        BandResult(c, None, None, rt60, None, 0.99) for c in (125, 500, 2000, 8000)
    ])


def reverberant_signal(rt60: float = 1.5, seconds: float = 4.0, seed: int = 1):
    """Dry click train convolved with a synthetic IR -> known reverberant signal."""
    from scipy.signal import fftconvolve
    sr = SR
    dry = np.zeros(int(seconds * sr))
    for k in range(4):
        dry[int((0.3 + k * 0.8) * sr)] = 1.0
    ir = synthetic_ir(rt60, seconds=2.0, seed=seed)
    wet = fftconvolve(dry, ir)[: len(dry)]
    return dry, wet


class TestDereverb:
    def test_zero_reduction_is_identity(self):
        _, wet = reverberant_signal()
        out = dereverb(wet, SR, make_profile(), DereverbSettings(reduction_db=0.0))
        assert out == pytest.approx(wet)

    def test_reduces_tail_energy(self):
        _, wet = reverberant_signal(rt60=1.5)
        out = dereverb(wet, SR, make_profile(1.5), DereverbSettings(reduction_db=12.0))
        # measure energy in a tail-only region (after last click + 200 ms)
        a = int((0.3 + 3 * 0.8 + 0.2) * SR)
        b = a + int(0.5 * SR)
        tail_in = float(np.sum(wet[a:b] ** 2))
        tail_out = float(np.sum(out[a:b] ** 2))
        assert tail_out < tail_in * 0.5  # at least 3 dB tail reduction

    def test_stereo_shape_and_image(self):
        _, wet = reverberant_signal()
        stereo = np.stack([wet, wet * 0.5], axis=1)
        out = dereverb(stereo, SR, make_profile(), DereverbSettings(reduction_db=8.0))
        assert out.shape == stereo.shape
        # same gain on both channels: channel ratio preserved
        mask = np.abs(out[:, 1]) > 1e-6
        ratio = out[mask, 0] / out[mask, 1]
        assert np.median(ratio) == pytest.approx(2.0, rel=0.05)

    def test_length_preserved(self):
        _, wet = reverberant_signal()
        out = dereverb(wet, SR, make_profile(), DereverbSettings(reduction_db=6.0))
        assert len(out) == len(wet)
