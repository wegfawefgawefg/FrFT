# FrFT

This is a scratch project for approximating audio with sine components (frequency + phase + power).

## What I was doing

- `src/rec.py` (and legacy `rec.py`): record 2 seconds of mic audio to `my_voice_recording.wav`.
- `src/one_freq.py` (and legacy `one_freq.py`): brute-force search for the best single sine wave that matches a target signal.
- `src/2_freq.py` (and legacy `2_freq.py`): load recorded voice, repeatedly find best sine components, subtract each component, then reconstruct/play/save (`best.wav`).

The "overall best frequency" output comes from this fitting loop, especially in `2_freq.py`.

## When (US/Central)

Most of this work happened early morning on **December 19, 2023**:

- `one_freq.py` modified: `2023-12-19 01:01:06 -0600`
- `test.py` modified: `2023-12-19 02:08:33 -0600`
- `rec.py` modified: `2023-12-19 03:21:27 -0600`
- `2_freq.py` modified: `2023-12-19 03:51:43 -0600`
- commit `a9d00fd` ("code"): `2023-12-19 03:55:53 -0600`

Initial repo commit:

- commit `1611017` ("Initial commit"): `2023-12-18 23:37:09 -0600`
