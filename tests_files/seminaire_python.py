






#on essaie deviter les iterations et les boucle pour eviter de jammer les execution pour rien

import numpy as np

import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile




def labo():
    for i in np.arange(0, 10):
        print(i)



x = np.asarray([0,1,2,1], dtype=np.float32)



# opening RAW files
note_guitare_LAd = './sounds/note_guitare_LAd.wav'
sampFreq, signal = wavfile.read(note_guitare_LAd)
x, Fs = sf.read('../sounds/note_guitare_LAd.wav')


plt.figure("Open .wav with wavfile from scipy.io'", figsize=(12, 6))
plt.title('RAW .wav file signal')
plt.plot(signal)
plt.show()


plt.figure("Open .wav with soundfile", figsize=(12, 6))
plt.title('RAW .wav file signal')
plt.plot(x)
plt.show()





#transformee de fourrier
X1signal = np.fft.fft(signal)
plt.figure("Open .wav with wavfile from scipy.io'", figsize=(12, 6))
plt.title('Transformee de Fourier')
plt.plot(X1signal)
plt.show()

#transformee de fourrier
X1 = np.fft.fft(x)
plt.figure("Open .wav with soundfile", figsize=(12, 6))
plt.title('Transformee de Fourier')
plt.plot(X1)
plt.show()






# ici ca va me donner la portion reelle mais seulement les premieres valeurs
# et ne repete pas les valeurs symetriques qui prennene tde dlespace pour rien dans notre anal;yse
#transformee de fourrier reelles
X2signal = np.fft.rfft(signal)
plt.figure("Open .wav with wavfile from scipy.io'", figsize=(12, 6))
plt.title('Transformee de Fourier Reelles')
plt.plot(X2signal)
plt.show()

#transformee de fourrier reelles
X2 = np.fft.rfft(x)
plt.figure("Open .wav with soundfile", figsize=(12, 6))
plt.title('Transformee de Fourier Reelles')
plt.plot(X2)
plt.show()




#mais il a ajoute des j mais il ne sait pas que cest juste reelle et met les imaginaires
#transformee de fourrier inverses
x1signal = np.fft.ifft(X1signal)
plt.figure("Open .wav with wavfile from scipy.io'", figsize=(12, 6))
plt.title('Transformee de Fourier Inverses')
plt.plot(x1signal)
plt.show()

#transformee de fourrier inverses
x1 = np.fft.ifft(X1)
plt.figure("Open .wav with soundfile", figsize=(12, 6))
plt.title('Transformee de Fourier Inverses')
plt.plot(x1)
plt.show()







#par contre irfft cest quon passe un spectre et le prog sait quil y a symetrie a lentree et il va effacer les composantres imaginaires et ne garde que le reelle
x2 = np.fft.irfft(X2)
#transformee de fourrier inverses reelles
x2 = np.fft.irfft(X1signal)
plt.figure("Open .wav with wavfile from scipy.io'", figsize=(12, 6))
plt.title('Transformee de Fourier Inverses reelles')
plt.plot(x2)
plt.show()

#transformee de fourrier inverses reelles
x1 = np.fft.irfft(X1)
plt.figure("Open .wav with soundfile", figsize=(12, 6))
plt.title('Transformee de Fourier Inverses reelles')
plt.plot(x1)
plt.show()




#maintenant quon veut faire convolution entre deux signaux on utilise la fonction convolt entre nos 2 signaux
x = np.asarray([0,1,2,1], dtype=np.float32)
h = np.asarray([3, -1, 2], dtype=np.float32)
y = np.convolve(x, h)










x, Fs = sf.read('../sounds/note_guitare_LAd.wav')

print(x)
print(Fs)
plt.plot(x)
#plt.show()







xx, Fss = sf.read('../sounds/note_basson_plus_sinus_1000_Hz.wav')
print(xx)
print(Fss)
plt.plot(xx)
# plt.show()





sf.write('../filtered_sounds/son_filtree.wav', x, samplerate=Fss)


def fct(x, gain=1.0):
    y = gain * x
    return y


x = np.asarray([1,2,4,5])
z=fct(x, gain=3.0)
y=x
print(x)
print(z)
print(y[2])



# avec numpy.copy(x) il cree un nouvel array et il ne modifierai pas x, ca agit comme des pointeurs
y=np.copy(x)

y[2]=10.0
print(x)
print(z)
print(y[2])






x = np.arange(0, 19)

# cree un fichier vecteur numpy
np.save('vecteur.npy', x)

#ceci sera utile se sauver amplitude, phase et amplitude pour ce quils demandent dans le rapport
#sauve en utf8 et donc cest tres leger sur le disque





