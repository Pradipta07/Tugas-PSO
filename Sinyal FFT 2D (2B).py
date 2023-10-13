# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:25:24 2023

@author: 62853
"""

import cmath
import numpy as np
import matplotlib.pyplot as plt

def fft2d(image):
    def fft1d(Y):
        N = len(Y)
        if N <= 1:
            return Y
        even = fft1d(Y[0::2])
        odd = fft1d(Y[1::2])
        T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

    rows, cols = image.shape
    # FFT on rows
    fft_rows = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        fft_rows[i, :] = fft1d(image[i, :])

    # FFT on columns
    fft_image = np.zeros((rows, cols), dtype=complex)
    for j in range(cols):
        fft_image[:, j] = fft1d(fft_rows[:, j])

    return fft_image

# Generate a sample 2D image
N = 256  # Image size
Y = np.linspace(0, 8 * np.pi, N)
X = np.linspace(0, 8 * np.pi, N)
Y, X = np.meshgrid(Y, X)
image = np.sin(np.sqrt(X**2 + Y**2))

# Apply the custom 2D FFT function
fft_custom = fft2d(image)

# Validate using NumPy's 2D FFT function
fft_numpy = np.fft.fft2(image)

# Plot the original image and FFT results
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(np.abs(fft_custom), cmap='gray')
plt.title('Custom 2D FFT')
plt.subplot(133)
plt.imshow(np.abs(fft_numpy), cmap='gray')
plt.title('NumPy 2D FFT')
plt.tight_layout()
plt.show()

# Mencetak hasil
print("Muhammad Athilla Pradipta Adjie")
print("5009211142")
