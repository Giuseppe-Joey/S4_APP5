



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
signal, fs = sf.read(sound_data)






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





def coupe_bande(fs,signal):
    N = 1024
    n = np.arange(1, N)
    n1 = np.arange(0, N)
    fc = 20

    #1
    m = fc * N / fs
    k = round((m * 2)+1)

    if (k % 2) == 0:
        k = k + 1


    else:
        print(" ")

    #2
    h_lp = np.zeros(N)
    h_lp[n] = (np.sin(np.pi * n * k / N)) / (N * np.sin(np.pi * n / N))
    h_lp[0] = k / N

    #3
    w1 = np.pi * ((k - 1) / N)
    w0 = np.pi - w1

    #4 dirac

    d = np.zeros(N)
    d[0] = 1

    #4
    h_cb = d * h_lp * 2 * np.cos(w0 * n1)

    #5
    sig_cb = np.convolve(h_cb, signal)

    plt.subplot(3, 1, 1)
    print(w1)
    plt.plot(signal)
    plt.title("signal")
    plt.subplot(3, 1, 2)
    print(w1)
    plt.plot(h_cb)
    plt.title("coupe-bande")
    plt.subplot(3, 1, 3)
    print(w1)
    plt.plot(sig_cb)
    plt.title("signal coupe")

    plt.tight_layout() #bien mettre les titres
    plt.show()






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
coupe_bande(fs,signal)