import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf

# je lis mes signaux de dep
sig1, fs1 = sf.read('note_guitare_LAd.wav')
sig2, fs2 = sf.read('note_basson_plus_sinus_1000_Hz.wav')


# creeation de la fennetre
w = np.hanning(sig1.size)
sig1w = sig1 * w
# sf.write('note2.wav',sig1w, samplerate=sig1.size)
# calculer fft

sig1wftt = np.fft.fft(sig1w)
modu_fft = np.abs(sig1wftt)
modu_db = 20 * np.log10(modu_fft)
phase_fft = np.angle(sig1wftt)

pics, _ = find_peaks(modu_db, height=-20, distance=1000, prominence=40)  # prominece=40)
table = pics[0:32]

# trouver les data
amp = modu_fft[table]
frequ = (table * fs1) / sig1.size
phase = phase_fft[table]

print(amp)
print()
print(frequ)
print()
print(phase)

t = np.arange(0, 160000)
sinsol = 0
sinmi = 0
sinfa = 0
sinre = 0
sinsilence = 0 * np.sin((2 * np.pi * t / fs1) + 1)

for i in range(0, 32):
    sinsol += modu_fft[i] * np.sin((2 * np.pi * frequ[i] * 0.841 * t / fs1) + phase[i])
    sinmi += modu_fft[i] * np.sin((2 * np.pi * frequ[i] * 0.707 * t / fs1) + phase[i])
    sinfa += modu_fft[i] * np.sin((2 * np.pi * frequ[i] * 0.749 * t / fs1) + phase[i])
    sinre += modu_fft[i] * np.sin((2 * np.pi * frequ[i] * 0.630 * t / fs1) + phase[i])

# finding k
W = np.pi / 1000
x = 1 / np.sqrt(2)
# for k in range(1000):
#    y = (np.sin(W*k/2))/(k*np.sin(W/2))
#    if (y>x-0.0001) and (y<x+0.0001):
#       print(k-1)
k = 885
h = np.ones(k) / k
y = np.convolve(h, np.abs(sig1))

sol = (sinsol * y[0:sinsol.size])/18
mi = (sinmi * y[0:sinmi.size])/18
fa = (sinfa * y[0:sinfa.size])/18
re = (sinre * y[0:sinre.size])/18
silence = sinsilence * y[0:sinre.size]

ta_shit = np.concatenate((sol, sol, sol, mi, silence, fa, fa, re))



sf.write('note3.wav', ta_shit, samplerate=fs1)

# print(x1)
plt.figure(1)
plt.plot(sig1)
plt.plot(y)

plt.figure(2)
plt.plot(sig1w)

plt.figure(3)
plt.subplot(4, 1, 1)
plt.plot(sol)
plt.subplot(4, 1, 2)
plt.plot(mi)
plt.subplot(4, 1, 3)
plt.plot(fa)
plt.subplot(4, 1, 4)
plt.plot(re)

#basson comence

K = 1024
H = np.ones(K) / K
L = np.zeros(K)
L[0] = 1
M = L - (2 * H * np.cos((2 * np.pi * 1000) / fs2))


# creeation de la fennetre
w2 = np.hanning(sig2.size)

sig2w = sig2 * w2

# sf.write('note2.wav',sig1w, samplerate=sig1.size)
# calculer fft

sig2wftt = np.fft.fft(sig2w)
modu_fft2 = np.abs(sig2wftt)
modu_db2 = 20 * np.log10(modu_fft2)
phase_fft2 = np.angle(sig2wftt)

pics2, _ = find_peaks(modu_db2, height=-20, distance=500, prominence=40)  # prominece=40)
table2 = pics2[0:32]

# trouver les data
amp2 = modu_fft2[table2]
frequ2 = (table2 * fs2) / sig2.size
phase2 = phase_fft2[table2]

t2 = np.arange(0, sig2.size)
sinsol2 = 0
sinmi2 = 0
sinfa2 = 0
sinre2 = 0
sinsilence2 = 0 * np.sin((2 * np.pi * t2 / fs2) + 1)

for i in range(0, 32):
    sinsol2 += modu_fft2[i] * np.sin((2 * np.pi * frequ2[i] * 0.841 * t2 / fs2) + phase2[i])
    sinmi2 += modu_fft2[i] * np.sin((2 * np.pi * frequ2[i] * 0.707 * t2 / fs2) + phase2[i])
    sinfa2 += modu_fft2[i] * np.sin((2 * np.pi * frequ2[i] * 0.749 * t2 / fs2) + phase2[i])
    sinre2 += modu_fft2[i] * np.sin((2 * np.pi * frequ2[i] * 0.630 * t2 / fs2) + phase2[i])

y2 = np.convolve(M, sig2)

y3=np.convolve(h, np.abs(y2))



sol2 = (sinsol2 * y3[0:sinsol2.size])/30/8
mi2 = (sinmi2 * y3[0:sinmi2.size])/30/8
fa2 = (sinfa2 * y3[0:sinfa2.size])/30/8
re2 = (sinre2 * y3[0:sinre2.size])/30/8
silence2 = sinsilence2 * y3[0:sinre2.size]

baassin = np.concatenate((sol2, sol2, sol2, mi2, silence2, fa2, fa2, re2))

sf.write('note4.wav', baassin, samplerate=fs2)

plt.figure(4)
plt.plot(sig2)
plt.plot(y2)
plt.plot(y3)

plt.figure(5)
plt.plot(sig2w)

plt.figure(6)
plt.subplot(4, 1, 1)
plt.plot(sol2)
plt.subplot(4, 1, 2)
plt.plot(mi2)
plt.subplot(4, 1, 3)
plt.plot(fa2)
plt.subplot(4, 1, 4)
plt.plot(re2)


plt.figure(7)
plt.plot(baassin)

plt.show()
# plt.show()

# how to do a fft

# plt.subplot(3,1,1)
# plt.plot(w)
# plt.subplot(3,1,2)
# plt.plot(np.abs(sig1))
# plt.subplot(3,1,3)
# plt.plot(np.abs(sig1w))
# plt.show()


# plt.show()


# on ajoute , 100 pour avoir plus d'echantillon sinon c'est juste 20
# X1=np.fft.fft(x1)
# X1w=np.fft.fft(x1w)
# print(x1)
# plt.plot(np.abs(x1))
# puissance=== 5**2
# vecteur de zero... x3=np.zeros(N)
# x3[]
# plt.plot(np.fft.fftshift((np.abs(sig1)+1e-10)))


# plt.plot(np.fft.fftshift((np.abs(sig1)+1e-10)))
# plt.plot(y)
# #plt.plot(np.fft.fftshift((np.abs(sig1w)+1e-10)))
# plt.show()


# x2=(-1)**n

# x2w=x2*w


# plt.subplot(3,1,1)
# plt.stem(w)
# plt.subplot(3,1,2)
# plt.stem(x2)
# plt.subplot(3,1,3)
# plt.plot(x2w)
# plt.show()


# X2=np.fft.fft(x2)
# X2w=np.fft.fft(x2w)

# plt.stem(np.abs(X2))
# plt.stem(np.abs(X2w))
# plt.show()


# x3=np.zeros(N)
# x3[10]=1
# x3w=x3*w

# plt.subplot(3,1,1)
# plt.stem(w)
# plt.subplot(3,1,2)
# plt.stem(x3)
# plt.subplot(3,1,3)
# plt.stem(x3w)
# plt.show()


# X3=np.fft.fft(x3)
# X3w=np.fft.fft(x3w)


# plt.plot(np.fft.fftshift(np.abs(X3)))
# plt.plot(np.fft.fftshift(np.abs(X3w)))
# plt.show()


# fc=2000hz
# fs=16000hz
# N=16
# k=5
# filtre
# question 2

# k=5
# m=64
# h=np.zeros(m)

# h[0]=k/m#

# h[1:m] = (np.sin(np.pi*n[1:m]*k/m))/(m*np.sin(np.pi*n[1:m]/m))

# print(h)
