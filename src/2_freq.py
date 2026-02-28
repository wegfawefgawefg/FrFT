import random
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
from tqdm import tqdm
import wavio


def rnd():
    return random.random() * 2 - 1


def energy(ys):
    return np.sum(np.abs(ys))


def merge_components(component_list, phase_tolerance=0.001):
    """
    Merges components with the same frequency and similar phases by summing their powers.

    :param component_list: List of tuples in the form (frequency, phase, power)
    :param phase_tolerance: The maximum difference in phase to consider two components as similar
    :return: List of tuples with merged components
    """
    merged = {}
    for freq, phase, power in component_list:
        # Check if we can merge with an existing component
        found_merge = False
        for (existing_freq, existing_phase), existing_power in merged.items():
            if freq == existing_freq and abs(phase - existing_phase) <= phase_tolerance:
                merged[(existing_freq, existing_phase)] += power
                found_merge = True
                break
        if not found_merge:
            merged[(freq, phase)] = power
    return [(freq, phase, power) for (freq, phase), power in merged.items()]


import enum


class Mode(enum.Enum):
    Generate = enum.auto()
    Load = enum.auto()


mode = Mode.Load

approximation_sample_rate = 20_000
sample_duration = 2  # seconds
time = np.linspace(0, sample_duration, approximation_sample_rate)

if mode == Mode.Generate:
    # params
    true_components = [
        # frequency, phase, power
        (144, 2 * np.pi * 0.0, 1.0),
        # (288, 2 * np.pi * 0.0, 1.0),
    ]

    # create target wave
    ys = np.zeros_like(time)
    for freq, phase, power in true_components:
        ys += np.sin(2 * np.pi * freq * time + phase) * power
elif mode == Mode.Load:
    loaded_file_sample_rate, high_res_ys = wavfile.read("my_voice_recording.wav")
    trim_samples = sample_duration * loaded_file_sample_rate
    high_res_ys = high_res_ys[:trim_samples]
    high_res_ys = high_res_ys / np.max(np.abs(high_res_ys))
    if len(high_res_ys.shape) > 1:  # if not mono
        high_res_ys = high_res_ys[:, 0]  # convert to mono
    ys = resample(high_res_ys, approximation_sample_rate)

# do the FrFT
NUM_ENERGY_REDUCTIONS = 10
low_freq = 20  # 20 Hz
high_freq = 8000  # 20 kHz
n = 1000  # Number of divisions

# Generate logarithmically spaced frequencies
frequency_sweep = np.logspace(np.log10(low_freq), np.log10(high_freq), n)

yst = ys.copy()
components = []
for n in tqdm(range(0, NUM_ENERGY_REDUCTIONS)):
    power = 1 / (2**n)

    best_overall_loss = float("inf")
    best_overall_frequency = None
    best_overall_phase = None
    for freq in frequency_sweep:
        best_phase_loss = float("inf")

        best_phase = 0
        phase = best_phase
        phase_delta = np.pi / 4  # Initial phase step
        delta_dir = 1

        for i in range(0, 40):
            if i != 0:
                phase = best_phase + delta_dir * phase_delta
                phase = phase % (2 * np.pi)

            test_wave = np.sin(2 * np.pi * freq * time + phase) * power
            loss = np.mean((yst - test_wave) ** 2)

            if loss < best_phase_loss:
                best_phase_loss = loss
                best_phase = phase
                phase_delta *= 0.5
            else:
                delta_dir *= -1

            if loss < best_overall_loss:
                best_overall_loss = loss
                best_overall_frequency = freq
                best_overall_phase = phase

    components.append((best_overall_frequency, best_overall_phase, power))
    wave = (
        np.sin(2 * np.pi * best_overall_frequency * time + best_overall_phase) * power
    )

    yst -= wave

components = merge_components(components)

for freq, phase, power in components:
    print(
        "Frequency: {}, Phase: {}, Power: {}".format(freq, phase / (2 * np.pi), power)
    )


def plot():
    # plotting
    best_wave = np.zeros_like(time)
    for freq, phase, power in components:
        best_wave += np.sin(2 * np.pi * freq * time + phase) * power
    plt.figure(figsize=(12, 6))
    plt.plot(time, ys, label="Original Wave")
    plt.plot(time, best_wave, label="Best Approximate Wave", linestyle="--")
    plt.title("Original vs Best Approximate Wave")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def play():
    # generate waves at higher sample rate for listening
    sample_rate = 44100  # 44100 Hz
    duration = 2  # 2 seconds
    high_res_time = np.linspace(0, duration, duration * sample_rate, endpoint=False)

    if mode == Mode.Generate:
        # the base wave
        high_res_generated_ys = np.zeros_like(high_res_time)
        for freq, phase, power in true_components:
            high_res_generated_ys += (
                np.sin(2 * np.pi * freq * high_res_time + phase) * power
            )
        sd.play(high_res_generated_ys, sample_rate)
        sd.wait()
    elif mode == Mode.Load:
        global high_res_ys
        global loaded_file_sample_rate
        sd.play(high_res_ys, loaded_file_sample_rate)
        sd.wait()

    # the best wave
    high_res_best_wave = np.zeros_like(high_res_time)
    for freq, phase, power in components:
        high_res_best_wave += np.sin(2 * np.pi * freq * high_res_time + phase) * power
    sd.play(high_res_best_wave, sample_rate)
    sd.wait()

    # save the best wave
    wavio.write("best.wav", high_res_best_wave, sample_rate, sampwidth=2)


plot()
play()
