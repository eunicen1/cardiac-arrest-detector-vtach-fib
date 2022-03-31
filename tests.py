import os
import pickle
from vfibTach import vfibTach
import numpy as np
import matplotlib.pyplot as plt

#predicts 0 for normal + 1 for VFIB/VTACH
data = # enter incoming signal in Volts (divide by mV if necessary)
data = data[:,1] #emg 1 if necessary otherwise comment out
n = len(data)
plt.plot(list(range(0, n)), data)
plt.xlabel("Progression of time (s)")
plt.ylabel("V")
plt.title("Normal neutral ECG recording")
plt.show()
p=vfibTach(data, n, 20, 0)
print(p)
