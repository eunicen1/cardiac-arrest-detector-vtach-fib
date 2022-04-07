import os
import neurokit2 as nk
from PQRSTf_2 import detectPQRSTf
from denoise import denoise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('ecgiddb-0') #change to normal ecg; t=20s

#dictionary of P, QRS, T, f, fP, fQRS, fT
sigD = {}
for norm in os.listdir():
    tfile = open(norm, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64(lines)
    arr = arr.reshape(arr.shape[0],)
    fullwave=arr[::2]
    xnorm = denoise(arr[::2]) #sampling rate = 500 whereas others are 250
    P, QRS, T = detectPQRSTf(xnorm, 20)
    sigD[norm+"_ecgiddb"] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        # START additional columns
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(fullwave))),6), # frequency domain

        round(np.float64(0),6),
        ]
print(sigD)
os.chdir('../cardially-1') #change to ca ecg; t=69s
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    tfile = open(ca, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64([ line.split() for line in lines])
    fullwave=arr
    arr = arr[:,1].reshape(arr.shape[0],)
    #69s -> first 20s
    N = arr.shape[0]
    first60mark = int((60/69)*N)
    arr = arr[:first60mark]
    xca = denoise(arr)
    P, QRS, T = detectPQRSTf(xca, 60)
    # plt.plot(abs(np.fft.fft(fullwave)))
    # plt.show()
    sigD[ca+"_cardially"] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        # START additional columns
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(fullwave))),6), # frequency domain
        # END additional columns

        round(np.float64(1),6),
        ]
print(sigD)
os.chdir('../cudb-1') #change to ca ecg; t=8mins
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    tfile = open(ca, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64([ line.split() for line in lines])
    fullwave=arr
    #8 mins -> 20s
    N = arr.shape[0]
    first5mmark = int((5*60/(8*60))*N)
    arr = arr[:first5mmark].reshape(first5mmark,)
    xca = denoise(arr)
    P, QRS, T = detectPQRSTf(xca, 60)
    sigD[ca+"_cudb"] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        # START additional columns
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(fullwave))),6), # frequency domain
        # END additional columns

        round(np.float64(1),6),
        ]
print(sigD)
os.chdir('../vfdb-1') #change to ca ecg; t=30mins
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    tfile = open(ca, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64([ line.split() for line in lines])
    fullwave=arr
    #30 mins -> 20s
    N = arr.shape[0]
    first5mmark = int((5*60/(30*60))*N)
    arr = arr[:first5mmark].reshape(first5mmark,)
    xca = denoise(arr)
    P, QRS, T = detectPQRSTf(xca, 60)
    sigD[ca+"_vfdb"] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(fullwave))),6), # frequency domain
        # END additional columns

        round(np.float64(1),6),
        ]
print(sigD)
os.chdir('../afdb-0') #change to normal ecg; t=10hrs
#dictionary of P, QRS, T, f, fP, fQRS, fT
for norm in os.listdir():
    tfile = open(norm, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64(lines)
    fullwave=arr
    #10 hrs -> 20s
    N = arr.shape[0]
    first5mmark = int((5*60/(10*60*60))*N)
    arr = arr[:first5mmark]
    xnorm = denoise(arr)

    # plt.plot(abs(np.fft.fft(fullwave)))
    # plt.show()

    P, QRS, T = detectPQRSTf(xnorm, 60)
    sigD[norm+"_afdb"] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(fullwave))),6), # frequency domain

        round(np.float64(0),6),
        ]

print(sigD)
sigPD = pd.DataFrame.from_dict(sigD).T
print(sigPD)
colnames = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW", 'y']
sigPD.columns = colnames
sigPD=sigPD.fillna(0)
sigPD.to_csv('../data_aggr_4.csv')

#cardiac arrest == 1
#normal == 0
