from cmath import exp
from cmath import pi

def diff(a):
    ret = []
    for i in range(len(a)-1):
        ret.append(a[i+1]-a[i])
    return ret

def dft(a):
    N = len(a)
    ret = [abs(exp(-2j*pi*complex(x)*complex(x)/N)*complex(x)) for x in a]
    return ret
