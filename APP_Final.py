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
    This function is a dictionnary containing all the k index,
    factor and frequency for every notes

    :return: dictionnary: dict: a dictionnary containing all musicals notes and values
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







def magnitude_to_dB(amplitude):
    """
    This function takes 1 argument(amplitude) and convert it into dB
    and print the input and output.

    :param magnitude: float: a signal amplitude to convert into dB
    :return: decibel: float: the argument passed converted to decibel
    """
    decibel = 20 * np.log10(amplitude)
    print("Converted {} into {} dB".format(amplitude, decibel))
    return decibel










def get_sound_print_details(sound_file):
    """
    This function takes 1 argument (sound file in .wav format) and takes the
    signal and print all key parameters of this file

    :param sound_file: file: the file in .wav format to open
    :return: signal: the signal extracted from the sound file
    """

    print("----------------------------------------")
    print("----- Sound processing function --------")
    print("----------------------------------------")
    signal, Fs = sf.read(sound_file)

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / Fs

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # print the sound type and the sample frequency
    print("Sound type is                : {}".format(signal.dtype))
    print("Sound sampFreq (Fs) is       : {}".format(Fs))
    print("Sound shape is               : {}".format(signal.shape))
    print("Sound length in secs is      : {}".format('{:,.3f}'.format(length_in_secs)))
    print("Sound time (what is this))   : {}".format(time))
    print("Sound size is                : {}".format(signal.size))
    print("----------------------------------------\n\n")


    # afficher le graphique du signal de base
    plt.title('Signal de Depart')
    plt.xlabel('Frequence Echantillonnage')
    plt.ylabel('Signal Amplitude')
    plt.plot(signal)
    plt.show()

    return signal
















# testing the function
magnitude_to_dB(10)
magnitude_to_dB(100)
magnitude_to_dB(1000)
magnitude_to_dB(10000)
magnitude_to_dB(100000)




# declaring a dictionnary
dictionnary = notes_frequency_dictionnary()
# for key, array in dictionnary.items():
#     print(key, array[0])



# ouvrir le fichier et afficher les details et le graph
sound_data = './sounds/note_guitare_LAd.wav'
signal = get_sound_print_details(sound_data)

















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
