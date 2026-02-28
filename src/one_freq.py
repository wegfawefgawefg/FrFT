import random
import matplotlib.pyplot as plt
import numpy as np


def rnd():
    return random.random() * 2 - 1


# Parameters
frequency = 7  # 1 Hz
time = np.linspace(0, 2, 1000)  # 2 seconds, with 1000 points
phase_offset = 2 * np.pi * 0.1  # 0.25 cycle phase offset

# Sin wave
ys = np.sin(2 * np.pi * frequency * time + phase_offset)

# # Plotting
# plt.plot(time, y)
# plt.title("1 Hz Sin Wave")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()


frequencies = range(1, 20)

best_overall_loss = float("inf")
best_overall_frequency = None
best_overall_phase = None

for freq in frequencies:
    best_phase_loss = float("inf")
    best_phase = 0
    phase = best_phase
    phase_delta = np.pi / 4  # Initial phase step
    delta_dir = 1

    for i in range(0, 40):
        if i != 0:
            phase = best_phase + delta_dir * phase_delta
            phase = phase % (2 * np.pi)

        test_wave = np.sin(2 * np.pi * freq * time + phase)
        loss = np.mean((ys - test_wave) ** 2)

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

print("Best frequency: {}".format(best_overall_frequency))
print("Best phase offset: {}".format(best_overall_phase / (2 * np.pi)))

# plotting
best_wave = np.sin(2 * np.pi * best_overall_frequency * time + best_overall_phase)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, ys, label="Original Wave")
plt.plot(time, best_wave, label="Best Approximate Wave", linestyle="--")
plt.title("Original vs Best Approximate Wave")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
