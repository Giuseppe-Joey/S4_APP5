




def sound_processing_filter_1000Hz(filename):
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
    sampFreq, signal = wavfile.read(filename)

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

    # plot the FFT amplitude before and after
    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, np.abs(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    plt.xlim(0, 1500)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')

    # filtering the 1000 Hz
    for i, f in enumerate(freq):
        if f < 1020 and f > 980:  # (1)
            fft_spectrum[i] = 0.0
        if f < 21 or f > 20000:  # (2)
            fft_spectrum[i] = 0.0

    plt.subplot(122)
    plt.stem(freq, np.abs(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('After filtering')
    plt.xlim(0, 1500)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()

    write_wav_file('note_basson_filtered.wav', sampFreq, length_in_secs, fft_spectrum)


# this function create a .wav file with random short integer bytes 99999 seconds duration
def write_wav_file(fileName, sampleFreq, duration, frequency):
    import wave, struct, math, random
    #sampleRate = 44100.0  # hertz
    #duration = 1.0  # seconds
    #frequency = 440.0  # hertz
    obj = wave.open(fileName, 'wb')
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleFreq)
    for i in range(int(duration)):
        #value = random.randint(-32767, 32767)
        data = struct.pack('<h', frequency)
        obj.writeframesraw(data)
    obj.close()