# File Name :               APP_Final.py
# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco - lomg2301
#                           Lucas Corrales - corl0701
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5




# 1 - appliquer un fenetrage
# 2 - faire une fft sur le signal
# 3 - extraire phase, amplitude et frequence des sinusoides principales
# 4 - redresser le signal (valeur absolue)
# 5 - faire un filtre passe bas RIF a coeficients egaux (gain DC \ 0dB)
# 6 - trouver lenveloppe temporelle
# 7 - trouver lordre N du signal




# 5 - utiliser le passe-bas et passer h[0] pour avoir passe bande
# 6 -











import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
import soundfile as sf






def notes_frequency_dictionnary():
    """
    This function is a dictionnary containing all the k index, factor and frequency for every notes

    :return: dictionnary
    """
    dictionnary = {'DO':    [-9, 0.595, 261.6],
                   'DO#':   [-8, 0.630, 277.2],
                   'RE':    [-7, 0.667, 293.7],
                   'RE#':   [-6, 0.707, 311.1],
                   'MI':    [-5, 0.749, 329.6],
                   'FA':    [-4, 0.794, 349.2],
                   'FA#':   [-3, 0.841, 370.0],
                   'SOL':   [-2, 0.891, 392.0],
                   'SOL#':  [-1, 0.944, 415.3],
                   'LA':    [0, 1.000, 440.0],
                   'LA#':   [1, 1.060, 466.2],
                   'SI':    [2, 1.123, 493.9]}
    return dictionnary




# declaring a dictionnary
dictionnary = notes_frequency_dictionnary()
# for key, array in dictionnary.items():
#     print(key, array[0])



def dB_to_Mag(decibel: int):

    return 0



# numero 1
#
# signal, Fs = sf.read('./sounds/note_guitare_LAd.wav')
#
#
# fft_spectrum = np.fft.rfft(signal)
# freq = np.fft.rfftfreq(signal.size, d=1. / Fs)
#
# # shifting up the signal
# fft_spectrum = np.abs(fft_spectrum)







# # Get magnitude and phase
# magnitude_max = 0
# magnitude = np.asarray(np.abs(fft_spectrum))
# peaks = find_peaks(fft_spectrum)
#
# # for i in enumerate(peaks):
# #     if peaks[i] > magnitude_max:
# #             magnitude_max = peaks[i]
#
# # print("peaks are: ", peaks.max())
# # for i in enumerate(magnitude):
# #     if magnitude[i] > magnitude_max:
# #         magnitude_max = magnitude[i]
# #
# # print(magnitude_max)
#
#
#
#
# phase = np.angle(fft_spectrum)
# print("Magnitude:", magnitude)
# np.set_printoptions(threshold=sys.maxsize)
# print("phase:", phase)
#
# # plot the FFT amplitude BEFORE
# plt.figure("Filtering a signal", figsize=(12, 6))
# plt.subplot(121)
# plt.stem(freq, fft_spectrum, 'b', markerfmt=" ", basefmt="-b")
# plt.title('Before filtering')
# plt.xlim(0, 5000)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('FFT Amplitude')



#plt.show()

def env_temp():
    signal, Fs = sf.read('./sounds/note_guitare_LAd.wav')
    #sigabs = np.abs(signal)


    plt.subplot(221)
    plt.plot(signal)
    plt.subplot(222)
    plt.plot(signal)
    plt.show()



env_temp()