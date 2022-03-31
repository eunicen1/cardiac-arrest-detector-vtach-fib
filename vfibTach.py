from denoise import denoise
from PQRSTf import detectPQRSTf
import numpy as np
import pickle

def vfibTach(data, origt_sec, adjt_sec=20, val=-1):
    model = pickle.load(open("vftach_pred.pkl", "rb"))
    ecg_den = denoise(data)
    N = data.shape[0]
    adjmark = int((adjt_sec/(origt_sec))*N)
    arr = np.copy(data)
    arr = arr[:adjmark]
    P, QRS, T, f, fs = detectPQRSTf(arr, adjt_sec) # change it up? recall 20, 60, 5mins
    sigD = np.array([
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        round(np.float64(f),6),
        round(np.float64(fs[0]),6),
        round(np.float64(fs[1]),6),
        round(np.float64(fs[2]),6),

        # START additional columns
        round(np.float64(P) +np.float64(QRS) ,6), #PQRS
        round(np.float64(QRS) + np.float64(T),6), #QRST
        round(np.argmax(abs(np.fft.fft(data))),6), # frequency domain
        ])

    sigD=np.nan_to_num(sigD, copy=True, nan=0.0, posinf=None, neginf=None)
    sigD = sigD.reshape(1,-1)
    prediction = model.predict(sigD)
    if prediction == 0:
        print("Normal signal detected.")
    else:
        print("Patient is in VFIB/VTACH!")
    if val != -1:
        y = np.array([round(np.float64(val),6)]).reshape(1,-1) #guessing
        return int(prediction==y)/1
    return -1
