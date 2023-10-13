from cmath import exp, pi
import matplotlib.pyplot as plt
import numpy as np

def my_fft(x):
    N = len(x)
    if N <= 1:
        return x
    # Rekursi
    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])
    # Bagian kedua dalam persamaan FFT
    q = [exp(-2j * pi * k / N) * odd[k] for k in range(N//2)]

    return \
        [ (even[k] + q[k]) for k in range(N//2)] + \
        [ (even[k] - q[k]) for k in range(N//2)]

# Parameter sinyal
A = 1
def signal1(t, A):
    return 1 if -A/2 < t < A/2 else 0
def signal2(t, A):
    return 1 if -A < t < A else 0
def signal3(t, A):
    return 1 if -3*A < t < 3*A else 0

# Parameter plot: n harus merupakan pangkat dua karena algoritma FFT menggunakan divide and conquer
t_interval = 7 * A
n = 512

# Generate interval waktu dari -t_interval/2 hingga t_interval/2
t = [i * t_interval / n for i in range(-n//2, n//2)]

signal1_data = [signal1(i, A) for i in t]
signal2_data = [signal2(i, A) for i in t]
signal3_data = [signal3(i, A) for i in t]

output_square = my_fft(signal1_data)
output_rectangle = my_fft(signal2_data)
output_wide = my_fft(signal3_data)

output_square_oneside = output_square[:n//2]
output_rectangle_oneside = output_rectangle[:n//2]
output_wide_oneside = output_wide[:n//2]

# Generate interval frekuensi dari 0 hingga n/2
f = list(range(n//2))

np_output_square = np.fft.fft(signal1_data)
np_output_rectangle = np.fft.fft(signal2_data)
np_output_wide = np.fft.fft(signal3_data)

np_output_square_oneside = np_output_square[:n//2]
np_output_rectangle_oneside = np_output_rectangle[:n//2]
np_output_wide_oneside = np_output_wide[:n//2]

plt.figure(figsize=(12, 8))

plt.subplot(331)
plt.plot(t, [signal1(i, A) for i in t])
plt.title('Fungsi Origin A/2')

plt.subplot(332)
plt.plot(f, output_square_oneside)
plt.title('Fungsi FFT A/2')

plt.subplot(333)
plt.plot(f, np_output_square_oneside)
plt.title('Fungsi NumPy FFT A/2')

plt.subplot(334)
plt.plot(t, [signal2(i, A) for i in t])
plt.title('Fungsi Origin A')

plt.subplot(335)
plt.plot(f, output_rectangle_oneside)
plt.title('Fungsi FFT A')

plt.subplot(336)
plt.plot(f, np_output_rectangle_oneside)
plt.title('Fungsi Numpy FFT A')

plt.subplot(337)
plt.plot(t, [signal3(i, A) for i in t])
plt.title('Fungsi Origin 3A')

plt.subplot(338)
plt.plot(f, output_wide_oneside)
plt.title('Fungsi FFT 3A')

plt.subplot(339)
plt.plot(f, np_output_wide_oneside)
plt.title('Fungsi Numpy FFT 3A')

fig = plt.gcf()
fig.subplots_adjust(hspace=0.4)
plt.show()

print ("FFT 1D-1A, 1B, 1C")
print ("Muhammad Athilla Pradipta Adjie")
print ("5009211142")