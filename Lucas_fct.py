



import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import hilbert, chirp
import sys
import soundfile as sf
# import sympy
# from sympy import symbols, solve
import math






#(extract 3 param)
#1 extraction des parametres
#2 applique fenetre de hanning
#3 fft + transformer en dB
#4 np.wear

#(ordre du filtre que tas besoin)
#coupe bas 20hz




sound_data = './sounds/note_guitare_LAd.wav'







def env_temp(sound_data,k):
    signal, fs = sf.read(sound_data)
    sigabs = np.abs(signal)

    h = np.ones(k)/k

    # filter the data using convolution
    sigh = np.convolve(h, sigabs)


    plt.subplot(311)
    plt.title("signal de base")
    plt.plot(signal)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.subplot(312)
    plt.title("signal redressé")
    plt.plot(sigabs)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.subplot(313)
    plt.title("enveloppe temporelle")
    plt.plot(sigh)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.tight_layout()
    plt.show()

    return sigh












def find_k():
    """
    This function
    :return:
    """

    k = 0
    while k < 1000:
        k = k + 0.01
        x = ((np.sin((np.pi / 1000) * k / 2)) / (k*(np.sin((np.pi / 1000) / 2)))) * np.sqrt(2)
        if (x > 0.99999) and (x < 1.00001):
            k = math.trunc(k)
            print("k value is: {}".format(k))
            return k




k = find_k()
env_temp(sound_data, k)