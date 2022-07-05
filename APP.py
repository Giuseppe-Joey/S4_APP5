





def notes_frequency_dictionnary():
    """
    This function is a dictionnary containing all the frequencies for every notes

    :return: dictionnary
    """
    dictionnary = { 'DO': 261.6,
                    'DO#': 277.2,
                    'RE': 293.7,
                    'RE#': 311.1,
                    'MI': 329.6,
                    'FA': 349.2,
                    'FA#': 370.0,
                    'SOL': 392.0,
                    'SOL#': 415.3,
                    'LA': 440.0,
                    'LA#': 466.2,
                    'SI': 493.9}
    return dictionnary




def notes_factor_dictionnary():
    """
    This function is a dictionnary containing all the frequencies for every notes

    :return: dictionnary
    """
    dictionnary = { 'DO': 0.595,
                    'DO#': 0.630,
                    'RE': 0.667,
                    'RE#': 0.707,
                    'MI': 0.749,
                    'FA': 0.794,
                    'FA#': 0.841,
                    'SOL': 0.891,
                    'SOL#': 0.944,
                    'LA': 1.000,
                    'LA#': 1.060,
                    'SI': 1.123}
    return dictionnary






def notes_K_index_dictionnary():
    """
    This function is a dictionnary containing all the k index for every notes

    :return: dictionnary
    """
    dictionnary = {'DO': (-9),
                   'DO#': (-8),
                   'RE': (-7),
                   'RE#': (-6),
                   'MI': (-5),
                   'FA': (-4),
                   'FA#': (-3),
                   'SOL': (-2),
                   'SOL#': (-1),
                   'LA': 0,
                   'LA#': 1,
                   'SI': 2}
    return dictionnary



def sound_processing_filter_1000Hz(filename_input, filename_output):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import wavfile

    print("----------------------------------------")
    print("----- Sound processing function --------")
    print("----------------------------------------")

    # matplotlib inline
    plt.style.use('seaborn-poster')

    # some magic to see better quality graphic
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)

    # reading the wav file (wavfile.read() function reads 16 or 32 bits wav file, 24 bits are not supported)
    sampFreq, signal = wavfile.read(filename_input)

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / sampFreq

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # print the sound type and the sample frequency
    print("Sound type is : {}".format(signal.dtype))
    print("Sound frequency is : {}".format(sampFreq))
    print("Sound shape is : {}".format(signal.shape))
    print("Sound length in secs is : {}".format('{:,.3f}'.format(length_in_secs)))
    print("Sound time (what is this)): {}".format(time))


    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / sampFreq)
    #print("Complex part:{}".format(fft_spectrum))

    #obtaining Amplitude vs Frequency spectrum we find absolute value of fourier transform
    fft_spectrum_abs = np.abs(fft_spectrum)

    # plot the FFT amplitude BEFORE
    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, fft_spectrum_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    plt.xlim(0, 1500)
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
    plt.xlim(0, 1500)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()



    rfft_spectrum_abs = np.fft.rfft(fft_spectrum_abs)
    ifft_spectrum_abs = np.fft.ifft(fft_spectrum_abs)
    irfft_spectrum = np.fft.irfft(fft_spectrum)
    irfft_spectrum_abs = np.fft.irfft(fft_spectrum_abs)
    fft_spectrum_abs = np.fft.fft(fft_spectrum_abs)

    # writing back the signal into .wav file
    wavfile.write(filename_output, sampFreq, irfft_spectrum_abs.astype(np.int16))





