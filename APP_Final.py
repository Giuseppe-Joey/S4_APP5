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










def sound_details(signal, Fs):
    """
    This function takes 1 argument (sound file in .wav format) and takes the
    signal and print all key parameters of this file

    :param sound_file: file: the file in .wav format to open
    :param Fs: file: the Sample Frequency
    :return: signal: the signal extracted from the sound file
    """

    print("----------------------------------------")
    print("----- Sound processing function --------")
    print("----------------------------------------")


    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / Fs

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # print the sound type and the sample frequency
    print("Sound dtype is                       : {}".format(signal.dtype))
    print("Sound sample frequency (Fs) is       : {}".format(Fs))
    print("Sound shape is (left(mono), right)   : {}".format(signal.shape))
    print("Sound length in secs is              : {}".format('{:,.3f}'.format(length_in_secs)))
    #print("Sound time (what is this))          : {}".format(time))
    print("Sound size is                        : {}".format(signal.size))
    print("----------------------------------------\n\n")

    # afficher le graphique du signal de base
    plt.title('Signal de Depart')
    plt.xlabel('Frequence Echantillonnage')
    plt.ylabel('Signal Amplitude')
    plt.plot(signal)
    plt.show()










def calculating_omega(Fs, f):
    omega = 2 * np.pi * f / Fs
    print("Fs       :{} Sample/secs".format(Fs))
    print("f        :{} Hz".format(f))
    print("Omega    :{} rad/echantillons".format('{:,.2f}'.format(omega)))

    return omega






def fft_show_real_imag(signal, Fs):

    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / Fs)

    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, np.real(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Partie Reelle')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')

    plt.subplot(122)
    plt.stem(freq, np.imag(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Partie Imaginaire')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()


    plt.show()

    return fft_spectrum







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
signal, Fs = sf.read(sound_data)
sound_details(signal, Fs)

f = 1000
omega = calculating_omega(Fs, f)



fft_show_real_imag(signal, Fs)












