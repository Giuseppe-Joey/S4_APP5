
def env_temp():
    signal, Fs = sf.read('./sounds/note_guitare_LAd.wav')
    #sigabs = np.abs(signal)


    plt.subplot(221)
    plt.plot(signal)
    plt.subplot(222)
    plt.plot(signal)
    plt.show()



env_temp()