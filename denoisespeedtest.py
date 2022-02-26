from denoise import denoise
from waveletDenoise import waveletDenoise as wdenoise
import os
from time import time
import numpy as np

os.chdir("ecgiddb")
tfile = open("Person_01.txt")

lines = (tfile.read().splitlines())

array = np.float64(lines)

start = time()
den1 = denoise(array)
end1 = time()-start

start = time()
den2 = wdenoise(array)
end2 = time()-start

print(end1, end2)
