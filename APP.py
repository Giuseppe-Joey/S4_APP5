

# File Name :               APP.py

# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco - lomg2301 && Lucas Corrales - corl0701
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP - testing before integrating to APP_Final.py






import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import soundfile as sf






def sound_processing_filter_1000Hz(filename_input, filename_output):


    print("----------------------------------------")
    print("----- Sound processing function --------")
    print("----------------------------------------")

    # matplotlib inline
    plt.style.use('seaborn-poster')

    # some magic to see better quality graphic
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)

    # reading the wav file (wavfile.read() function reads 16 or 32 bits wav file, 24 bits are not supported)
    #sampFreq, signal = wavfile.read(filename_input)
    signal, sampFreq = sf.read(filename_input)

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / sampFreq

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / sampFreq)
    print("Freq:{}".format(freq))

    #obtaining Amplitude vs Frequency spectrum we find absolute value of fourier transform
    fft_spectrum_abs = np.abs(fft_spectrum)

     # plot the FFT amplitude BEFORE
    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, fft_spectrum_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')



    # filtering the 1000 Hz
    for i, f in enumerate(freq):
        if f < 1020 and f > 980:  # (1)
            fft_spectrum[i] = 0.0
            fft_spectrum_abs[i] = 0.0

    # plot the FFT amplitude AFTER
    plt.subplot(122)
    plt.stem(freq, fft_spectrum_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.title('After filtering')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()

    irfft_spectrum = np.fft.irfft(fft_spectrum)

    # writing back the signal into .wav file
    wavfile.write(filename_output, sampFreq, irfft_spectrum.astype(np.int16))





def apply_fft_with_window(signal):

    # creating a window from the Fs (Sample Frequency)
    window = np.hanning(signal.size)
    print("This is the window:", window)

    # keeping only the real numbers and not the complex part
    signal = signal * window
    fft_spectrum = np.fft.fft(signal)
    fft_spectrum = np.fft.fftshift(np.log(np.abs(fft_spectrum)))
    freq = np.fft.rfftfreq(fft_spectrum.size, d=1. / signal.size)

    print("Freq:{}".format(freq))

    return fft_spectrum






# Running the function
note_basson_1000_Hz = './sounds/note_basson_plus_sinus_1000_Hz.wav'
note_basson_filtered = './filtered_sounds/note_basson_filtered.wav'

sound_processing_filter_1000Hz(note_basson_1000_Hz, note_basson_filtered)
















# numero 1
# signal, Fs = sf.read('./sounds/note_guitare_LAd.wav')
#
#
# fft_spectrum = np.fft.rfft(signal)
# freq = np.fft.rfftfreq(signal.size, d=1. / Fs)
#
# # shifting up the signal
# fft_spectrum = np.abs(fft_spectrum)







# Get magnitude and phase
# magnitude_max = 0
# magnitude = np.asarray(np.abs(fft_spectrum))
# peaks = find_peaks(fft_spectrum)

# for i in enumerate(peaks):
#     if peaks[i] > magnitude_max:
#             magnitude_max = peaks[i]

# print("peaks are: ", peaks.max())
# for i in enumerate(magnitude):
#     if magnitude[i] > magnitude_max:
#         magnitude_max = magnitude[i]
#
# print(magnitude_max)




# phase = np.angle(fft_spectrum)
# print("Magnitude:", magnitude)
# np.set_printoptions(threshold=sys.maxsize)
# print("phase:", phase)

# # plot the FFT amplitude BEFORE
# plt.figure("Filtering a signal", figsize=(12, 6))
# plt.subplot(121)
# plt.stem(freq, fft_spectrum, 'b', markerfmt=" ", basefmt="-b")
# plt.title('Before filtering')
# plt.xlim(0, 5000)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('FFT Amplitude')



# plt.show()
