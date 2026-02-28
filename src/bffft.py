import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brute-force FFT-like analyzer: for each time bin and frequency bin, "
            "sweep phase and fit magnitude."
        )
    )
    parser.add_argument("wav_path", type=Path, help="Path to an input WAV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bffft_image.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=40.0,
        help="Window length in milliseconds",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
        help="Hop length in milliseconds",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=20.0,
        help="Lowest frequency bin (Hz)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=8000.0,
        help="Highest frequency bin (Hz)",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        default=128,
        help="Number of frequency bins",
    )
    parser.add_argument(
        "--phase-steps",
        type=int,
        default=32,
        help="Number of phases to test per frequency",
    )
    parser.add_argument(
        "--scale",
        choices=["log", "linear"],
        default="log",
        help="Frequency axis scaling",
    )
    parser.add_argument(
        "--greedy-iters",
        type=int,
        default=1,
        help=(
            "Greedy residual subtraction iterations per frame. "
            "1 means no subtraction, >1 means iterative component peeling."
        ),
    )
    return parser.parse_args()


def to_float_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[:, 0]

    if np.issubdtype(audio.dtype, np.integer):
        max_int = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_int
    else:
        audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def frame_signal(audio: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    if len(audio) < window_size:
        pad = window_size - len(audio)
        audio = np.pad(audio, (0, pad))

    starts = np.arange(0, len(audio) - window_size + 1, hop_size)
    frames = np.stack([audio[s : s + window_size] for s in starts], axis=0)
    return frames


def build_templates(
    sample_rate: int,
    window_size: int,
    freqs: np.ndarray,
    phase_steps: int,
    window_fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(window_size, dtype=np.float32) / sample_rate
    phases = np.linspace(0.0, 2.0 * np.pi, phase_steps, endpoint=False, dtype=np.float32)

    omega_t = 2.0 * np.pi * freqs[:, None] * t[None, :]
    templates = np.sin(omega_t[:, None, :] + phases[None, :, None])
    templates *= window_fn[None, None, :]

    denom = np.sum(templates * templates, axis=2)
    denom = np.maximum(denom, 1e-12)
    return templates.astype(np.float32), denom.astype(np.float32), phases


def brute_force_image(
    frames: np.ndarray,
    templates: np.ndarray,
    denom: np.ndarray,
    greedy_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = frames.shape[0]
    num_freqs = templates.shape[0]
    image = np.zeros((num_freqs, num_frames), dtype=np.float32)
    phase_map = np.zeros((num_freqs, num_frames), dtype=np.float32)

    for frame_idx in range(num_frames):
        residual = frames[frame_idx].copy()
        for _ in range(greedy_iters):
            # Correlation with every (freq, phase) template.
            corr = np.tensordot(templates, residual, axes=([2], [0]))
            mags = corr / denom

            best_phase_idx = np.argmax(np.abs(mags), axis=1)
            row_idx = np.arange(num_freqs)
            best_mags = mags[row_idx, best_phase_idx]

            image[:, frame_idx] += np.abs(best_mags)
            phase_map[:, frame_idx] = best_phase_idx

            if greedy_iters > 1:
                best_freq_idx = int(np.argmax(np.abs(best_mags)))
                best_phase = int(best_phase_idx[best_freq_idx])
                amp = best_mags[best_freq_idx]
                residual = residual - amp * templates[best_freq_idx, best_phase]

    return image, phase_map


def save_image(
    image: np.ndarray,
    freqs: np.ndarray,
    sample_rate: int,
    hop_size: int,
    output_path: Path,
) -> None:
    image_db = 20.0 * np.log10(np.maximum(image, 1e-8))
    image_db -= np.max(image_db)

    num_frames = image.shape[1]
    duration_s = (num_frames * hop_size) / sample_rate

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    times = np.arange(num_frames + 1, dtype=np.float32) * (hop_size / sample_rate)

    # Build frequency bin edges so plotting is correct for linear and log-spaced bins.
    if len(freqs) == 1:
        f0 = float(freqs[0])
        freq_edges = np.array([max(1e-3, f0 * 0.95), f0 * 1.05], dtype=np.float32)
    else:
        mids = 0.5 * (freqs[:-1] + freqs[1:])
        low = max(1e-3, freqs[0] - (mids[0] - freqs[0]))
        high = freqs[-1] + (freqs[-1] - mids[-1])
        freq_edges = np.concatenate(([low], mids, [high])).astype(np.float32)

    plt.pcolormesh(
        times,
        freq_edges,
        image_db,
        shading="auto",
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    if np.all(np.diff(freqs) > 0):
        ratio = freqs[-1] / max(freqs[0], 1e-12)
        if ratio > 10:
            plt.yscale("log")
    plt.colorbar(label="Relative magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("BFFFT (Brute-Force Frequency/Phase Sweep)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()

    sample_rate, raw_audio = wavfile.read(args.wav_path)
    audio = to_float_mono(raw_audio)

    nyquist = sample_rate * 0.5
    fmax = min(args.fmax, nyquist - 1.0)
    if fmax <= args.fmin:
        raise ValueError(
            f"Invalid frequency range: fmin={args.fmin}, fmax={fmax}, nyquist={nyquist}"
        )

    if args.scale == "log":
        freqs = np.logspace(np.log10(args.fmin), np.log10(fmax), args.freq_bins).astype(
            np.float32
        )
    else:
        freqs = np.linspace(args.fmin, fmax, args.freq_bins, dtype=np.float32)

    window_size = max(16, int(sample_rate * (args.window_ms / 1000.0)))
    hop_size = max(1, int(sample_rate * (args.hop_ms / 1000.0)))

    window_fn = np.hanning(window_size).astype(np.float32)
    frames = frame_signal(audio, window_size, hop_size) * window_fn[None, :]

    templates, denom, _ = build_templates(
        sample_rate=sample_rate,
        window_size=window_size,
        freqs=freqs,
        phase_steps=args.phase_steps,
        window_fn=np.ones(window_size, dtype=np.float32),
    )

    image, phase_map = brute_force_image(
        frames=frames,
        templates=templates,
        denom=denom,
        greedy_iters=max(1, args.greedy_iters),
    )

    save_image(
        image=image,
        freqs=freqs,
        sample_rate=sample_rate,
        hop_size=hop_size,
        output_path=args.output,
    )

    np.save(args.output.with_suffix(".magnitude.npy"), image)
    np.save(args.output.with_suffix(".phase_idx.npy"), phase_map)

    print(f"WAV: {args.wav_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frames: {frames.shape[0]}, window: {window_size}, hop: {hop_size}")
    print(f"Freq bins: {len(freqs)}, phase steps: {args.phase_steps}")
    print(f"Saved image: {args.output}")
    print(f"Saved magnitude matrix: {args.output.with_suffix('.magnitude.npy')}")
    print(f"Saved phase index matrix: {args.output.with_suffix('.phase_idx.npy')}")


if __name__ == "__main__":
    main()
