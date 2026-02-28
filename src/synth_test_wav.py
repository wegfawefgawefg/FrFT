import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean multi-sine WAV for BFFFT validation"
    )
    parser.add_argument(
        "--freqs",
        type=str,
        default="220,440,880",
        help="Comma-separated frequencies in Hz (e.g. 220,440,880)",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz")
    parser.add_argument(
        "--phase-mode",
        choices=["zero", "random"],
        default="zero",
        help="Use aligned or random starting phases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ideal_multi_tone.wav"),
        help="Output WAV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freqs = [float(x.strip()) for x in args.freqs.split(",") if x.strip()]
    if not freqs:
        raise ValueError("No frequencies provided")

    t = np.arange(int(args.duration * args.sample_rate), dtype=np.float32) / args.sample_rate

    rng = np.random.default_rng(0)
    phases = np.zeros(len(freqs), dtype=np.float32)
    if args.phase_mode == "random":
        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs)).astype(np.float32)

    y = np.zeros_like(t)
    for f, p in zip(freqs, phases):
        y += np.sin(2.0 * np.pi * f * t + p)

    # Equal-volume components and headroom-safe normalization.
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.95 * (y / peak)

    out_i16 = (y * 32767.0).astype(np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, args.sample_rate, out_i16)

    print(f"Saved: {args.output}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Duration: {args.duration:.3f} s")
    print(f"Frequencies: {freqs}")
    print(f"Phase mode: {args.phase_mode}")


if __name__ == "__main__":
    main()
