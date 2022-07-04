
#code source :::: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html


import matplotlib.pyplot as plt
import numpy as np



#---------------------------------------
#generating a signal
#---------------------------------------

# matplotlib inline
plt.style.use('seaborn-poster')

# sampling rate
sr = 2000

# sampling interval
ts = 1.0 / sr
t = np.arange(0, 1, ts)

freq = 1.
x = 3 * np.sin(2 * np.pi * freq * t)

freq = 4
x += np.sin(2 * np.pi * freq * t)

freq = 7
x += 0.5 * np.sin(2 * np.pi * freq * t)

plt.figure("generating a signal", figsize=(8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()
#---------------------------------------











#---------------------------------------
#using FFT
#---------------------------------------
from numpy.fft import fft, ifft

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure("using FFT", figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
#---------------------------------------


# 36.2 µs ± 775 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)










#---------------------------------------
#using FFT in Scipy
#---------------------------------------
from scipy.fftpack import fft, ifft



X = fft(x)

plt.figure("Using FFT in Scipy", figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
#---------------------------------------










#---------------------------------------
#Filtering a signal using FFT
#---------------------------------------
from scipy.fftpack import fftfreq

plt.figure("Filtering a signal", figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.title('Original signal')
plt.show()


# FFT the signal
sig_fft = fft(x)
# copy the FFT results
sig_fft_filtered = sig_fft.copy()

# obtain the frequencies using scipy function
freq = fftfreq(len(x), d=1./2000)

# define the cut-off frequency
cut_off = 6

# high-pass filter by assign zeros to the
# FFT amplitudes where the absolute
# frequencies smaller than the cut-off
sig_fft_filtered[np.abs(freq) < cut_off] = 0

# get the filtered signal in time domain
filtered = ifft(sig_fft_filtered)

# plot the filtered signal
plt.figure("Filtering a signal", figsize = (12, 6))
plt.plot(t, filtered)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# plot the FFT amplitude before and after
plt.figure("Filtering a signal", figsize = (12, 6))
plt.subplot(121)
plt.stem(freq, np.abs(sig_fft), 'b', markerfmt=" ", basefmt="-b")
plt.title('Before filtering')
plt.xlim(0, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Amplitude')
plt.subplot(122)
plt.stem(freq, np.abs(sig_fft_filtered), 'b', markerfmt=" ", basefmt="-b")
plt.title('After filtering')
plt.xlim(0, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Amplitude')
plt.tight_layout()
plt.show()