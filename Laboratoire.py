
# File Name :               Laboratoire.py

# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5 - Laboratoire



import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf








def exercice1():
    # definir mon vecteur avec n seulement
    # le N est la fenetre, plus la fenetre est grande, plus on a une bonne resolution
    N = 200 # aleatoire pour commencer
    n = np.arange(0, N)
    print(n)




    # creation dune fenetre de Hann
    #   IMPORTANT::::: LA FENETRE DOIT TJRS ETRE MULTIPLIE SUR LE SIGNAL AVANT DE FAIRE LA FFT
    w = np.hanning(N)
    print(w)





    # il sagit du signal 1 demande dans le labo
    x1 = np.sin((0.1 * np.pi * n) + (np.pi / 4))
    print(x1)
    plt.plot(x1)
    plt.show()



    # ensuite on multiplie le signal par la fenetre de Hann
    x1w = x1*w
    plt.plot(x1w)
    plt.show()

    X1w = np.fft.fft(x1w)
    plt.plot(X1w)
    plt.show()






    # ici on faity la fft et le 2e argument est le nombre de point quon lui dit de prendre
    X1 = np.fft.fft(x1)
    plt.plot(X1)
    plt.show()





    plt.subplot(3, 1, 1)
    plt.title("Fenetre de Hann")
    plt.plot(w)
    plt.subplot(3, 1, 2)
    plt.title("Sans Fenetre de Hann")
    plt.plot(x1)
    plt.subplot(3, 1, 3)
    plt.title("Avec Fenetre de Hann")
    plt.plot(X1)
    plt.subplot(3, 1, 4)
    plt.title("Avec Fenetre de Hann")
    plt.plot(x1w)
    plt.show()


    # now with the real value only and with fftshift
    X1 = np.fft.fftshift(np.log(np.abs(X1)))
    print(X1)
    plt.plot(X1)
    plt.show()



    # le signal 2 est donne dans le labo
    # en python quand on veut faire un exposant on fait **
    x2 = (-1) ** n
    print(x2)

    X2 = np.fft.fft(x2)
    plt.title("Signal x2")
    plt.plot(X2)
    plt.show()





    # le signal 3 on doit faire un dirac
    dirac = 0

    # avec np.zero on genere des 0
    x3 = np.zeros(N)

    # ensuite si on veut le 10e terme on ecrit
    x3[10] = 10


    dirac = 1
    x3 = 0


exercice1()





# ********************************************
# probleme B :::: CONCEPTION DES  FILTRES FIR
# ********************************************
#on doit choisir les coeficients pour que le filtre ai un coeficient particulier
#on veut concevoir un filtre (voir le N, le Fc, et le Fs dans le labo)


# quand on veut concevoir on commence par conevoir un passe bas et ensuite on converti en coupe, etc
# 1 -  on defini lordre du filtre, ici 16, 32, 64
# 2 - on choisi la frequence de coupure, ici 2000 Hz
# 3 - puis on choisi la frequence dechantillonnage ici 16000hz
# 4 -  cependant on veut la frequence dechantillonnage en radian / echantillons
#--> on sait que le 2pi est aligne avec la Fs, le pi est aligne a la Fs/2 et le 0 avec 0
# ce quon veut cest la frequence transforme en radian par echantillon, on sait que 16000 = 2*pi donc
# on fait une regle de trois et donc 2000*2pi / 16000 = pi/4

#maintenant on veut la valeur de k qui est un nombre entier et qui va nous donner une expression qui va se rappocher le plus de, pi / 4
# donc pi*( K - 1 / 16 ) soit aprox egal a pi/4 *******ici le 16 vient du N quon a defini au debut
#donc K ici serait egal a 5
#ICI SUPER IMPORTANT DE NE PAS GARDER LES COEFICIENTS PAIRS, ON VEUT Garder SEULEMENT LES IMPAIRS

# maintenant on peut determiner les coeficients pour calculer la reponse impulsionnelle du filtre, la formule est donnee dans le guide etudiant h[n]

#donc selon la formule elle va nous donner les coeficients du filtre
#on obtiendra donc un signal cardinal


# pour un cas en particulier on aura un probleme , quand le n=0 on aura une division par 0 donc
# on devra utiliser la regle de lhopital pour n=0, qui dit que l<on doit faire la deriver du haut et la derivee du bas

# donc la derivee du haut est --> cos(pi*n*K/N) *pi*K/N le tout divise par Ncos(pi*n/N)*pi/N
# donc quand n=0 , la fonction sera h[0] = K/N, on nfera donc un if()

# une fois quon a trouve les coeficients, pour appliquer les coeficients on doit faire une convolution et lapplique au signal donc on aura le signal filtre dans le domaine du temps
# cela sera donc un filtre passe bas qui eliminera donc les frequence en haut de 2000 Hz

# la derniere chose que pour cet exemple on aura pas a faire mais on pourrait aussi transformer notre passe-bas en PASSE-BANDE!!!!!







H = []


def filtre_FIR(list_h):

    N = [16, 32, 64, 256, 512, 1024]
    Fs = 16000  # frequence dechantillon
    Fc = 2000  # frequence de coupure
    K = np.asarray([0, 0, 0, 0, 0, 0, 0], dtype=np.int16)      # list of coeficients

    # ici cest la freq en radians par echantillons
    w = Fc * 2 * np.pi / Fs
    print(w)

    # ici on va aller multiplier chacune des valeur de N par Fs en rad par echantillons
    # for i, value in enumerate(N):
    #     # we want to make sure we keep only the impair values
    #     if int(((w / np.pi) * value) + 1) % 2 != 0:
    #         K[i] = ((w / np.pi) * value) + 1
    #         print("Value of N[{}]: {} and value of K[{}]: {}".format(i, value, i, K[i]))



    #when n=0

    #h = []



    N=1024
    n = np.arange(0, N)


    for i, value in enumerate(n):
        K = ((w / np.pi) * value) + 1
        if (value == 0):
            list_h.append((np.cos(np.pi*value*K/N) * np.pi*K/N) / (N* np.cos(np.pi*value/N)*np.pi/N))
            #print(list_h)
        else:
            list_h.append(np.sin(np.pi * value * K / N) / N*np.sin(np.pi*value/N))
            #print(list_h)





# filtre_FIR(H)

# plt.plot(H)
# plt.show()



