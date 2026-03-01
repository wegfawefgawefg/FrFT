# BFFFT Notes

This document captures what `src/bffft.py` is doing and the core concepts behind it.

## Intent

The project intent was to rediscover FFT-like analysis from scratch by sweeping sine masks across audio:

- sweep frequencies
- sweep phases
- fit magnitudes
- build a time-frequency image

This is a brute-force spectral estimator, not a canonical FFT implementation.

## High-level Pipeline

`src/bffft.py` does:

1. Load WAV, convert to mono float, normalize peak.
2. Split waveform into overlapping time frames.
3. Build a bank of sine templates over `(frequency, phase, time-in-frame)`.
4. For each frame, correlate against all templates.
5. For each frequency, pick best phase and fitted magnitude.
6. Store magnitudes in a spectrogram-like 2D image.
7. Save:
- image (`.png`)
- magnitude matrix (`.magnitude.npy`)
- chosen phase index matrix (`.phase_idx.npy`)

## Key Arrays and Shapes

Let:

- `F = num_freq_bins`
- `P = phase_steps`
- `T = window_size`
- `M = num_frames`

Then:

- `freqs`: shape `(F,)`
- `t`: shape `(T,)` where `t = sample_index / sample_rate`
- `omega_t`: shape `(F, T)` with `2*pi*freq*t`
- `templates`: shape `(F, P, T)` with `sin(omega_t + phase)`
- `frames`: shape `(M, T)`
- `corr`: shape `(F, P)` for one frame
- `mags`: shape `(F, P)` for one frame
- `best_mags`: shape `(F,)` for one frame
- `image`: shape `(F, M)` final magnitude map

## Why Divide by Sample Rate in `t`

`np.arange(window_size)` is sample index, not seconds.

Sine model uses real time:

`sin(2*pi*f*t + phase)`

So:

`t = n / sample_rate`

Without dividing by sample rate, frequencies are scaled incorrectly.

## What `phases` Is

`phases` is an evenly spaced list from `0` to `< 2*pi`, length `phase_steps`.

Each phase is a candidate alignment for a given frequency. The code picks the best phase per frequency per frame.

## What `denom` Is

For a frame `y` and one template `s`, amplitude is fitted by least squares:

`a = (y dot s) / (s dot s)`

`denom = s dot s = sum(s^2)` for each template.

Why needed:

- raw dot product is scale-biased by template energy
- dividing by `s dot s` gives a proper amplitude estimate

Equivalent alternative: normalize templates to unit norm ahead of time and skip denominator at runtime.

## Window Function (Hann)

Frames are tapered with Hann window before analysis to reduce edge discontinuities and leakage.

Windowing is standard in STFT-like analysis.

## Is This Similar to Neural Nets?

Conceptually yes:

- template bank resembles fixed filters/features
- dot products resemble pre-activations
- per-frequency best-phase selection resembles max/argmax pooling
- normalization controls scale

Difference: no learned weights/backprop here; templates are analytic sinusoids.

## Sign vs Phase

For sine, sign and phase are related:

`-sin(x) = sin(x + pi)`

So signed amplitude and phase can encode overlapping information.

Conventions to avoid ambiguity:

- `A >= 0`, phase in `[0, 2*pi)`, or
- signed `A`, phase in fixed interval

## Redundancy Notes

There is representational redundancy in sine-only `(A, phase)` parameterization:

- `A*sin(wt+phi)`
- `(-A)*sin(wt+phi+pi)`
- phase wraps by `2*pi`

Using sin+cos projections (or complex coefficients) gives cleaner non-redundant handling of phase.

## Why Sin+Cos Is Better Than Phase Sweep

If you project onto both bases for each frequency:

- `c = y dot cos(wt)`
- `s = y dot sin(wt)`

Then:

- magnitude = `sqrt(c^2 + s^2)`
- phase = `atan2(s, c)`

This avoids discrete phase grid quantization and is closer to standard DFT/FFT math.

## Does Greedy Iterative Version Work?

Yes, it can work, but it is approximate.

Greedy residual subtraction (`--greedy-iters > 1`) can recover dominant components but may:

- misattribute energy
- smear nearby components
- accumulate local-optimum errors

Still valid as an exploratory decomposition method.

## Complexity and Practicality

Compared with FFT:

- FFT: `O(N log N)` per frame scale
- brute force here: roughly `O(F * P * T)` per frame

So BFFFT is usually slower than FFT for general use.

Where BFFFT is useful:

- custom/nonuniform frequency grids
- explicit phase search experiments
- prototyping alternative spectral ideas

## Parallelism

Computation across `(frame, frequency, phase)` combinations is highly parallel.

This is embarrassingly parallel and can map well to GPU/FPGA architectures, subject to template memory and bandwidth constraints.

## Reconstruction (Inverse Direction)

You do not need one model per sample.

Use framewise coefficients and overlap-add:

1. For each frame and frequency, keep coefficient + phase (or equivalent signed form).
2. Reconstruct each frame from summed sinusoidal components.
3. Overlap-add frames at hop offsets.
4. Normalize by overlap window weights.

Important:

- absolute magnitudes are fine for visualization
- signed/phase-consistent coefficients are needed for good waveform reconstruction

## Current Validation Status

An ideal synthetic multi-tone test (e.g., `220/440/880 Hz`) showed dominant detected peaks near expected bins and produced the expected horizontal-line structure in the image.

So the implementation is behaving as a rough FFT-like analyzer.

## Bottom Line

BFFFT is a legit brute-force spectral analysis approach and a good exploration tool. It is not better than FFT for standard transform efficiency, but it is useful for understanding and experimenting with frequency/phase decomposition from first principles.
