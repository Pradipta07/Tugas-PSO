# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:40:51 2023

@author: 62853
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate a sample signal
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
freq = 5  # Frequency of the signal (Hz)
signal_waveform = np.sin(2 * np.pi * freq * t)

# Add noise to the signal
noise = np.random.normal(0, 0.5, fs)
noisy_signal = signal_waveform + noise

# Design a low-pass Butterworth filter
cutoff_frequency = 10  # Cutoff frequency (Hz)
order = 4  # Filter order
b, a = signal.butter(order, cutoff_frequency / (fs / 2), 'low')

# Apply the filter to the noisy signal
filtered_signal = signal.lfilter(b, a, noisy_signal)

# Perform FFT on the filtered signal
fft_result = np.fft.fft(filtered_signal)
frequencies = np.fft.fftfreq(len(fft_result), 1/fs)

# Create a modulation signal
modulation_frequency = 2  # Modulation frequency (Hz)
modulation_signal = np.sin(2 * np.pi * modulation_frequency * t)

# Perform convolution between the filtered signal and the modulation signal
convolution_result = np.convolve(filtered_signal, modulation_signal, 'same')

# Plot the original, noisy, and filtered signals
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal_waveform)
plt.title("Original Signal")

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal)
plt.title("Noisy Signal")

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal)
plt.title("Filtered Signal")

plt.tight_layout()

# Plot the FFT result
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(fft_result))
plt.title("FFT of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot the convolution result
plt.figure(figsize=(12, 6))
plt.plot(t, convolution_result)
plt.title("Convolution Result")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.show()

