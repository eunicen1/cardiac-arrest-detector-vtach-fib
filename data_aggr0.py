import os
import neurokit2 as nk

from waveletDenoise import waveletDenoise as denoise
from calc import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp

os.chdir('ecgiddb-0') #change to normal ecg; t=20s

#dictionary of P, QRS, T, f, fP, fQRS, fT
sigD = {}
for norm in os.listdir():
    try:
        tfile = open(norm, "r")
        lines = (tfile.read().splitlines())
        arr = list(lines)
        xnorm = denoise(arr)
        _, rpeaks = nk.ecg_peaks(xnorm, sampling_rate=500)
        R = rpeaks['ECG_R_Peaks']
        amp_R = xnorm[R]
        diff_R = diff(R)
        avg_f_R = sum(diff_R)/len(diff_R) #feature
        avg_amp_R = sum(amp_R)/len(amp_R) #feature
        _, waves_peak = nk.ecg_delineate(xnorm, rpeaks, sampling_rate=500, method="peak")
        T = waves_peak['ECG_T_Peaks']
        amp_T = xnorm[T]
        diff_T = diff(T)
        avg_f_T = sum(diff_T)/len(diff_T) #feature
        avg_amp_T = sum(amp_T)/len(amp_T) #feature
        P = waves_peak['ECG_P_Peaks']
        amp_P = xnorm[P]
        diff_P = diff(P)
        avg_f_P = sum(diff_P)/len(diff_P) #feature
        avg_amp_P = sum(amp_P)/len(amp_P) #feature
        Q = waves_peak['ECG_Q_Peaks']
        amp_Q = xnorm[Q]
        diff_Q = diff(Q)
        avg_f_Q = sum(diff_Q)/len(diff_Q) #feature
        avg_amp_Q = sum(amp_Q)/len(amp_Q) #feature
        S = waves_peak['ECG_S_Peaks']
        amp_S = xnorm[S]
        diff_S = diff(S)
        avg_f_S = sum(diff_S)/len(diff_S) #feature
        avg_amp_S = sum(amp_S)/len(amp_S) #feature
        dft_res = dft(xnorm)
        sigD[norm+"_ecgiddb"] = [
            round(avg_f_R,6),
            round(avg_amp_R,6),
            round(avg_f_T,6),
            round(avg_amp_T,6),
            round(avg_f_Q,6),
            round(avg_amp_Q,6),
            round(avg_f_S,6),
            round(avg_amp_S,6),
            round(dft_res.index(max(dft_res)),6), # frequency domain
            round(np.float64(0),6),
            ]
        print("done: ", sigD[norm+"_ecgiddb"])
    except:
        print("Something oopsed. Continuing anyways.")
        continue

os.chdir('../cardially-1') #change to ca ecg; t=69s
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()

for ca in files:
    try:
        tfile = open(norm, "r")
        lines = (tfile.read().splitlines())
        arr = list(lines)
        xca = denoise(arr)
        _, rpeaks = nk.ecg_peaks(xca, sampling_rate=50)
        R = rpeaks['ECG_R_Peaks']
        amp_R = xca[R]
        diff_R = diff(R)
        avg_f_R = sum(diff_R)/len(diff_R) #feature
        avg_amp_R = sum(amp_R)/len(amp_R) #feature
        _, waves_peak = nk.ecg_delineate(xca, rpeaks, sampling_rate=50, method="peak")
        T = waves_peak['ECG_T_Peaks']
        amp_T = xca[T]
        diff_T = diff(T)
        avg_f_T = sum(diff_T)/len(diff_T) #feature
        avg_amp_T = sum(amp_T)/len(amp_T) #feature
        P = waves_peak['ECG_P_Peaks']
        amp_P = xca[P]
        diff_P = diff(P)
        avg_f_P = sum(diff_P)/len(diff_P) #feature
        avg_amp_P = sum(amp_P)/len(amp_P) #feature
        Q = waves_peak['ECG_Q_Peaks']
        amp_Q = xca[Q]
        diff_Q = diff(Q)
        avg_f_Q = sum(diff_Q)/len(diff_Q) #feature
        avg_amp_Q = sum(amp_Q)/len(amp_Q) #feature
        S = waves_peak['ECG_S_Peaks']
        amp_S = xca[S]
        diff_S = diff(S)
        avg_f_S = sum(diff_S)/len(diff_S) #feature
        avg_amp_S = sum(amp_S)/len(amp_S) #feature
        dft_res = dft(xca)
        sigD[ca+"_cardially"] = [
            round(avg_f_R,6),
            round(avg_amp_R,6),
            round(avg_f_T,6),
            round(avg_amp_T,6),
            round(avg_f_Q,6),
            round(avg_amp_Q,6),
            round(avg_f_S,6),
            round(avg_amp_S,6),
            round(dft_res.index(max(dft_res)),6), # frequency domain
            round(np.float64(1),6),
            ]
        print("done: ", sigD[ca+"_cardially"])
    except:
        print("Something oopsed. Continuing anyways.")
        continue
os.chdir('../cudb-1') #change to ca ecg; t=8mins
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    try:
        tfile = open(norm, "r")
        lines = (tfile.read().splitlines())
        arr = list(lines)
        xca = denoise(arr)
        _, rpeaks = nk.ecg_peaks(xca, sampling_rate=250)
        R = rpeaks['ECG_R_Peaks']
        amp_R = xca[R]
        diff_R = diff(R)
        avg_f_R = sum(diff_R)/len(diff_R) #feature
        avg_amp_R = sum(amp_R)/len(amp_R) #feature
        _, waves_peak = nk.ecg_delineate(xca, rpeaks, sampling_rate=250, method="peak")
        T = waves_peak['ECG_T_Peaks']
        amp_T = xca[T]
        diff_T = diff(T)
        avg_f_T = sum(diff_T)/len(diff_T) #feature
        avg_amp_T = sum(amp_T)/len(amp_T) #feature
        P = waves_peak['ECG_P_Peaks']
        amp_P = xca[P]
        diff_P = diff(P)
        avg_f_P = sum(diff_P)/len(diff_P) #feature
        avg_amp_P = sum(amp_P)/len(amp_P) #feature
        Q = waves_peak['ECG_Q_Peaks']
        amp_Q = xca[Q]
        diff_Q = diff(Q)
        avg_f_Q = sum(diff_Q)/len(diff_Q) #feature
        avg_amp_Q = sum(amp_Q)/len(amp_Q) #feature
        S = waves_peak['ECG_S_Peaks']
        amp_S = xca[S]
        diff_S = diff(S)
        avg_f_S = sum(diff_S)/len(diff_S) #feature
        avg_amp_S = sum(amp_S)/len(amp_S) #feature
        dft_res = dft(xca)
        sigD[ca+"_cudb"] = [
            round(avg_f_R,6),
            round(avg_amp_R,6),
            round(avg_f_T,6),
            round(avg_amp_T,6),
            round(avg_f_Q,6),
            round(avg_amp_Q,6),
            round(avg_f_S,6),
            round(avg_amp_S,6),
            round(dft_res.index(max(dft_res)),6), # frequency domain
            round(np.float64(1),6),
            ]
        print("done: ", sigD[ca+"_cudb"])
    except:
        print("Something oopsed. Continuing anyways.")
        continue
os.chdir('../vfdb-1') #change to ca ecg; t=30mins
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    try:
        tfile = open(norm, "r")
        lines = (tfile.read().splitlines())
        arr = list(lines)
        xca = denoise(arr)
        _, rpeaks = nk.ecg_peaks(xca, sampling_rate=250)
        R = rpeaks['ECG_R_Peaks']
        amp_R = xca[R]
        diff_R = diff(R)
        avg_f_R = sum(diff_R)/len(diff_R) #feature
        avg_amp_R = sum(amp_R)/len(amp_R) #feature
        _, waves_peak = nk.ecg_delineate(xca, rpeaks, sampling_rate=250, method="peak")
        T = waves_peak['ECG_T_Peaks']
        amp_T = xca[T]
        diff_T = diff(T)
        avg_f_T = sum(diff_T)/len(diff_T) #feature
        avg_amp_T = sum(amp_T)/len(amp_T) #feature
        P = waves_peak['ECG_P_Peaks']
        amp_P = xca[P]
        diff_P = diff(P)
        avg_f_P = sum(diff_P)/len(diff_P) #feature
        avg_amp_P = sum(amp_P)/len(amp_P) #feature
        Q = waves_peak['ECG_Q_Peaks']
        amp_Q = xca[Q]
        diff_Q = diff(Q)
        avg_f_Q = sum(diff_Q)/len(diff_Q) #feature
        avg_amp_Q = sum(amp_Q)/len(amp_Q) #feature
        S = waves_peak['ECG_S_Peaks']
        amp_S = xca[S]
        diff_S = diff(S)
        avg_f_S = sum(diff_S)/len(diff_S) #feature
        avg_amp_S = sum(amp_S)/len(amp_S) #feature
        dft_res = dft(xca)
        sigD[ca+"_vfdb"] = [
            round(avg_f_R,6),
            round(avg_amp_R,6),
            round(avg_f_T,6),
            round(avg_amp_T,6),
            round(avg_f_Q,6),
            round(avg_amp_Q,6),
            round(avg_f_S,6),
            round(avg_amp_S,6),
            round(dft_res.index(max(dft_res)),6), # frequency domain
            round(np.float64(1),6),
            ]
        print("done: ", sigD[ca+"_vfdb"])
    except:
        print("Something oopsed. Continuing anyways.")
        continue
os.chdir('../afdb-0') #change to normal ecg; t=10hrs
#dictionary of P, QRS, T, f, fP, fQRS, fT
for norm in os.listdir():
    try:
        tfile = open(norm, "r")
        lines = (tfile.read().splitlines())
        arr = list(lines)
        xnorm = denoise(arr)
        _, rpeaks = nk.ecg_peaks(xnorm, sampling_rate=250)
        R = rpeaks['ECG_R_Peaks']
        amp_R = xnorm[R]
        diff_R = diff(R)
        avg_f_R = sum(diff_R)/len(diff_R) #feature
        avg_amp_R = sum(amp_R)/len(amp_R) #feature
        _, waves_peak = nk.ecg_delineate(xnorm, rpeaks, sampling_rate=250, method="peak")
        T = waves_peak['ECG_T_Peaks']
        amp_T = xnorm[T]
        diff_T = diff(T)
        avg_f_T = sum(diff_T)/len(diff_T) #feature
        avg_amp_T = sum(amp_T)/len(amp_T) #feature
        P = waves_peak['ECG_P_Peaks']
        amp_P = xnorm[P]
        diff_P = diff(P)
        avg_f_P = sum(diff_P)/len(diff_P) #feature
        avg_amp_P = sum(amp_P)/len(amp_P) #feature
        Q = waves_peak['ECG_Q_Peaks']
        amp_Q = xnorm[Q]
        diff_Q = diff(Q)
        avg_f_Q = sum(diff_Q)/len(diff_Q) #feature
        avg_amp_Q = sum(amp_Q)/len(amp_Q) #feature
        S = waves_peak['ECG_S_Peaks']
        amp_S = xnorm[S]
        diff_S = diff(S)
        avg_f_S = sum(diff_S)/len(diff_S) #feature
        avg_amp_S = sum(amp_S)/len(amp_S) #feature
        dft_res = dft(xnorm)
        sigD[norm+"_afdb"] = [
            round(avg_f_R,6),
            round(avg_amp_R,6),
            round(avg_f_T,6),
            round(avg_amp_T,6),
            round(avg_f_Q,6),
            round(avg_amp_Q,6),
            round(avg_f_S,6),
            round(avg_amp_S,6),
            round(dft_res.index(max(dft_res)),6), # frequency domain
            round(np.float64(0),6),
            ]
        print("done: ", sigD[norm+"_afdb"])
    except:
        print("Something oopsed. Continuing anyways.")
        continue

sigPD = pd.DataFrame.from_dict(sigD).T
sigPD=sigPD.fillna(0)
print(sigPD)
sigPD.to_csv('../data_aggr_fin.csv')

#cardiac arrest == 1
#normal == 0
