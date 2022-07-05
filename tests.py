
import matplotlib.pyplot as plt
import numpy as np

from numpy.fft import fft, ifft

from scipy.fftpack import fft, ifft, fftfreq


import numpy as np



def TEST1():
    n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_range = np.arange(0,20) + 1

    print("This is a range written totally", n)
    print("This is a range written with np.arange() function", n_range)








def wav_file_parameters(file):

    import wave
    # rb is for reading and wb is for writing
    obj = wave.open(file,'rb')
    print( "Number of channels",obj.getnchannels())
    print ( "Sample width",obj.getsampwidth())
    print ( "Frame rate.",obj.getframerate())
    print ("Number of frames",obj.getnframes())
    print ( "parameters:",obj.getparams())
    obj.close()






# this function create a .wav file with random short integer bytes 99999 seconds duration
def write_wav_file_test(fileName):
    import wave, struct, math, random
    sampleRate = 44100.0  # hertz
    duration = 1.0  # seconds
    frequency = 440.0  # hertz
    obj = wave.open(fileName, 'wb')
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)
    for i in range(99999):
        value = random.randint(-32767, 32767)
        data = struct.pack('<h', value)
        obj.writeframesraw(data)
    obj.close()







# analyze .wav file with matplotlib....not working!!!
def analyze_wav_file(filename):

    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    from scipy.io import wavfile  # get the api
    fs, data = wavfile.read(filename)  # load the data
    a = data.T[0]  # this is a two channel soundtrack, I get the first track
    b = [(ele / 2 ** 8.) * 2 - 1 for ele in a]  # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b)  # calculate fourier transform (complex numbers list)
    d = len(c) / 2  # you only need half of the fft list (real signal symmetry)
    plt.plot(abs(c[:(d - 1)]), 'r')
    plt.show()





# analyze wav files with fft
def analyze_wav_file_2(filename):

    import matplotlib.pyplot as plt
    from scipy.io import wavfile as wav
    from scipy.fftpack import fft
    import numpy as np

    rate, data = wav.read(filename)
    fft_out = fft(data)

    # matplotlib inline
    plt.style.use('seaborn-poster')
    plt.plot(data, np.abs(fft_out))
    plt.show()








def sound_processing_test(filename):
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
    sampFreq, sound = wavfile.read(filename)

    # getting the duration in seconds from frequency
    length_in_secs = sound.shape[0] / sampFreq

    # print the sound type and the sample frequency
    print("Sound type is : {}".format(sound.dtype))
    print("Sound frequency is : {}".format(sampFreq))
    print("Sound shape is : {}".format(sound.shape))
    print("Sound length in secs is : {}".format('{:,.3f}'.format(length_in_secs)))


    # plotting the sound on each channel
    # plt.subplot(2, 1, 1)
    # plt.plot(sound, 'r')
    # plt.xlabel("left channel, sample #")
    # comment the following section if the file is mono
    # plt.subplot(2, 1, 2)
    # plt.plot(sound[:, 1], 'b')
    # plt.xlabel("right channel, sample #")
    # plt.tight_layout()
    #plt.show()

    # creating an array containing the time for the x_axis
    time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_secs

    # plotting the sound on each channel
    # plt.subplot(2, 1, 1)
    # plt.plot(time, sound, 'r')
    # plt.xlabel("time, secs [left channel]")
    # comment the following section if the file is mono
    # plt.subplot(2, 1, 2)
    # plt.plot(sound[:, 1], 'b')
    # plt.xlabel("right channel, sample #")
    #plt.tight_layout()
    # plt.show()

    #looking at the sound with higher resolution
    signal = sound
    # plt.plot(time[6000:7000], signal[6000:7000])
    # plt.xlabel("time, s")
    # plt.ylabel("Signal, relative units")
    #plt.show()

    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / sampFreq)
    print("Complex part:{}".format(fft_spectrum))

    #obtaining Amplitude vs Frequency spectrum we find absolute value of fourier transform
    fft_spectrum_abs = np.abs(fft_spectrum)

    # the spectrum of the sound in the frequency domain
    # plt.plot(freq, fft_spectrum_abs)
    # plt.xlabel("frequency, Hz")
    # plt.ylabel("Amplitude, units")
    # plt.show()

    # zoom on the highest peaks
    # plt.plot(freq[:4000], fft_spectrum_abs[:4000])
    # plt.xlabel("frequency, Hz")
    # plt.ylabel("Amplitude, units")
    # plt.show()

    # zoom in even more
    # plt.plot(freq[:500], fft_spectrum_abs[:500])
    # plt.xlabel("frequency, Hz")
    # plt.ylabel("Amplitude, units")
    # plt.arrow(90, 5500, -20, 1000, width=2, head_width=8, head_length=200, fc='k', ec='k')
    # plt.arrow(200, 4000, 20, -1000, width=2, head_width=8, head_length=200, fc='g', ec='g')
    # plt.show()

    # look at the peaks more truly
    # for i, f in enumerate(fft_spectrum_abs):
    #     if f > 350 and f < 2000:  # looking at amplitudes of the spikes higher than 350 and lower than 2000
    #         print('frequency = {} Hz with amplitude {} '.format(np.round(freq[i], 1), np.round(f)))

    # plot the FFT amplitude before and after
    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, np.abs(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    plt.xlim(0, 2000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')


    # filtering the 1000 Hz
    for i, f in enumerate(freq):
        if f < 1020 and f > 980:  # (1)
            fft_spectrum[i] = 0.0
        if f < 21 or f > 20000:  # (2)
            fft_spectrum[i] = 0.0

    # plt.plot(freq[:4000], np.abs(fft_spectrum[:4000]))
    # plt.xlabel("frequency, Hz")
    # plt.ylabel("Amplitude, units")
    # plt.show()





    plt.subplot(122)
    plt.stem(freq, np.abs(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('After filtering')
    plt.xlim(0, 2000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()

    print("----------------------------------------")
