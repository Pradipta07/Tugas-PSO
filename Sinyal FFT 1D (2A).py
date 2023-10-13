# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:12:12 2023

@author: 62853
"""

import cmath
import numpy as np
import matplotlib.pyplot as plt

def fft(H):
    N = len(H)
    if N <= 1:
        return H
    even = fft(H[0::2])
    odd = fft(H[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Generate a sample 1D signal
N = 64  # Number of data points
T = 1.0 / 800.0  # Sample spacing
H = np.linspace(0.0, N * T, N)
y = np.sin(50.0 * 2.0 * np.pi * H) + 0.5 * np.sin(80.0 * 2.0 * np.pi * H)

# Apply the custom FFT function
H_custom = fft(y)

# Validate using NumPy's FFT function
H_numpy = np.fft.fft(y)

# Plot the original signal, custom FFT, and NumPy FFT
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(H, y)
plt.title('Original Signal')
plt.subplot(312)
plt.plot(np.abs(H_custom))
plt.title('Custom FFT')
plt.subplot(313)
plt.plot(np.abs(H_numpy))
plt.title('NumPy FFT')
plt.tight_layout()
plt.show()

# Mencetak hasil
print("Muhammad Athilla Pradipta Adjie")
print("5009211142")

