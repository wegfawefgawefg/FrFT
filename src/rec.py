import sounddevice as sd
import numpy as np
import wavio

# Set the duration and sample rate
duration = 2  # seconds
sample_rate = 44100  # Hz

# Record audio for the set duration
print("Recording...")
myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Save the recording to a WAV file
wavio.write("my_voice_recording.wav", myrecording, sample_rate, sampwidth=2)
print("Recording saved to 'my_voice_recording.wav'.")
