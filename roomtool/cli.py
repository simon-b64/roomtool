"""Command-line interface for roomtool.

Commands:
    roomtool analyze <ir.txt|ir.wav>      -> room_profile.json + analysis plot
    roomtool dereverb <in.wav> --profile room_profile.json --reduction 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def cmd_analyze(args: argparse.Namespace) -> int:
    import soundfile as sf  # noqa: F401  (fail early if missing)

    from .rew_import import load_impulse_response
    from .analysis import analyze, plot_analysis

    ir = load_impulse_response(args.input, sample_rate=args.sr)
    print(f"Loaded IR: {len(ir.samples)} samples @ {ir.sample_rate:.0f} Hz "
          f"({ir.duration:.2f} s), peak at index {ir.peak_index}")

    profile = analyze(ir)

    print(f"\n{'Band':>8}  {'EDT':>6}  {'T20':>6}  {'T30':>6}  {'RT60':>6}  {'fit r²':>7}")
    for b in profile.bands:
        def fmt(v):
            return f"{v:.2f}" if v is not None else "  -- "
        r2 = b.t30_r2 if b.t30 is not None else b.t20_r2
        print(f"{b.center_hz:>6.0f}Hz  {fmt(b.edt):>6}  {fmt(b.t20):>6}  "
              f"{fmt(b.t30):>6}  {fmt(b.rt60):>6}  {fmt(r2):>7}")

    if profile.drr_db is not None:
        print(f"\nDirect-to-reverberant ratio: {profile.drr_db:+.1f} dB")
    if profile.c50_db is not None:
        print(f"Clarity C50: {profile.c50_db:+.1f} dB   C80: {profile.c80_db:+.1f} dB")

    out_json = Path(args.output or "room_profile.json")
    profile.save(out_json)
    print(f"\nProfile written to {out_json}")

    plot_path = out_json.with_suffix(".png")
    plot_analysis(ir, profile, plot_path)
    print(f"Plot written to {plot_path}")
    return 0


def cmd_dereverb(args: argparse.Namespace) -> int:
    import soundfile as sf

    from .analysis import RoomProfile
    from .dereverb import dereverb, DereverbSettings

    profile = RoomProfile.load(args.profile)
    audio, sr = sf.read(args.input, dtype="float64", always_2d=False)
    info = sf.info(args.input)
    print(f"Loaded {args.input}: {info.frames} frames, {info.channels} ch @ {sr} Hz")

    reductions = args.sweep or [args.reduction]
    settings_base = dict(
        late_start_ms=args.late_start,
        window=args.window,
    )

    processed = audio
    if args.wpe:
        from .wpe_stage import wpe_dereverb

        print("Running WPE pre-stage ...")
        processed = wpe_dereverb(processed, sr)

    in_path = Path(args.input)
    for red in reductions:
        settings = DereverbSettings(reduction_db=red, **settings_base)
        print(f"Dereverberating at {red:g} dB reduction ...")
        out = dereverb(processed, sr, profile, settings)

        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak * 0.999
            print(f"  output peaked at {peak:.2f}, normalized down")

        suffix = f"_dereverb{red:g}dB" + ("_wpe" if args.wpe else "")
        out_path = Path(args.output) if args.output and len(reductions) == 1 \
            else in_path.with_name(in_path.stem + suffix + ".wav")
        sf.write(out_path, out, int(sr), subtype=info.subtype if "FLOAT" in info.subtype else "PCM_24")
        print(f"  written: {out_path}")

        if args.residual:
            n = min(len(audio), len(out))
            res = np.asarray(audio)[:n] - np.asarray(out)[:n]
            res_path = out_path.with_name(out_path.stem + "_residual.wav")
            sf.write(res_path, res, int(sr), subtype="PCM_24")
            print(f"  residual: {res_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roomtool",
                                     description="Room IR analysis and RIR-informed dereverberation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyze a REW impulse response export")
    p_an.add_argument("input", help="REW IR export (.txt with header, or .wav)")
    p_an.add_argument("-o", "--output", help="Output profile JSON (default: room_profile.json)")
    p_an.add_argument("--sr", type=float, help="Sample rate for headerless text files")
    p_an.set_defaults(func=cmd_analyze)

    p_de = sub.add_parser("dereverb", help="Reduce reverberation in a recording")
    p_de.add_argument("input", help="Input WAV recording (mono or stereo)")
    p_de.add_argument("--profile", required=True, help="room_profile.json from 'analyze'")
    p_de.add_argument("--reduction", type=float, default=6.0,
                      help="Maximum reverb reduction in dB (default: 6)")
    p_de.add_argument("--sweep", type=float, nargs="+", metavar="DB",
                      help="Render several reduction amounts in one run, e.g. --sweep 3 6 10 15")
    p_de.add_argument("--late-start", type=float, default=60.0,
                      help="Late reverb onset in ms (default: 60)")
    p_de.add_argument("--window", type=int, default=4096,
                      help="STFT window size (default: 4096)")
    p_de.add_argument("--wpe", action="store_true",
                      help="Run blind WPE pre-stage first (requires nara-wpe)")
    p_de.add_argument("--residual", action="store_true",
                      help="Also write input-minus-output WAV to hear what was removed")
    p_de.add_argument("-o", "--output", help="Output WAV path (single reduction only)")
    p_de.set_defaults(func=cmd_dereverb)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
