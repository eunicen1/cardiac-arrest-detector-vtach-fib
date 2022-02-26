import wfdb
import os
import numpy as np

# os.chdir('ecgiddb')
# files = os.listdir()
# for norm in files:
#     name_norm = norm + "/rec_1"
#     record_norm = wfdb.rdrecord(name_norm)
#     xnorm = np.array(record_norm.__dict__['p_signal'][:, 0]) #raw signal
#     x = xnorm.shape[0]
#     xnorm = xnorm.reshape(x,1)
#     os.chdir('../ecgiddb2')
#     tfile = open(norm+".txt", "w") #assume rec_1
#     for row in xnorm:
#         np.savetxt(tfile, row)
#     tfile.close()
#     os.chdir('../ecgiddb')

os.chdir('afdb-0')
files = os.listdir()
for avtc in files:
    name_avtc = avtc
    record_avtc = wfdb.rdrecord(name_avtc)
    xavtc = np.array(record_avtc.__dict__['p_signal'][:, 0]) #raw signal
    x = xavtc.shape[0]
    xavtc = xavtc.reshape(x,1)
    os.chdir('../afdb-0-2')
    tfile = open(avtc+".txt", "w") #assume rec_1
    for row in xavtc:
        np.savetxt(tfile, row)
    tfile.close()
    os.chdir('../afdb-0')

os.chdir('cudb-1')
files = os.listdir()
for cudb in files:
    name_cudb = cudb
    record_cudb = wfdb.rdrecord(name_cudb)
    xcudb = np.array(record_cudb.__dict__['p_signal'][:, 0]) #raw signal
    x = xcudb.shape[0]
    xcudb = xcudb.reshape(x,1)
    os.chdir('../cudb-1-2')
    tfile = open(cudb+".txt", "w") #assume rec_1
    for row in xcudb:
        np.savetxt(tfile, row)
    tfile.close()
    os.chdir('../cudb-1')

os.chdir('mvtdb-1')
files = os.listdir()
for mvtdb in files:
    name_mvtdb = mvtdb
    record_mvtdb = wfdb.rdrecord(name_mvtdb)
    xmvtdb = np.array(record_mvtdb.__dict__['p_signal'][:, 0]) #raw signal
    x = xmvtdb.shape[0]
    xmvtdb = xmvtdb.reshape(x,1)
    os.chdir('../mvtdb-1-2')
    tfile = open(mvtdb+".txt", "w") #assume rec_1
    for row in xmvtdb:
        np.savetxt(tfile, row)
    tfile.close()
    os.chdir('../mvtdb-1')
