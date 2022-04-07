import pywt
from pywt import dwt
from os import chdir
import numpy as np
from matplotlib import pyplot as plt
from denoise import denoise

def waveletDenoise(arr):
    #print(pywt.wavelist(kind='discrete'))
    return (dwt(arr, 'bior1.5')[0])

def waveletDenoiseTest(arr):
    wvlist = []
    for type in list(pywt.wavelist(kind='discrete')):
        wvlist.append(dwt(arr, type)[0])
    return wvlist
