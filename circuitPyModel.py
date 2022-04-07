import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve as pltroc
from sklearn.metrics import precision_recall_curve as pltprc
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MaxAbsScaler
from time import time
import jsonpickle

import statsmodels.formula.api as smf

np.random.seed(457)

#setup
#START added
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW", 'y', '']
df = pd.read_csv('data_aggr_4.csv', delimiter=',', header=None, names=col_names)
del df[df.columns[0]]
scaler = MaxAbsScaler()
scaler.fit(df)
scaled = scaler.transform(df)
df = pd.DataFrame(scaled, columns=df.columns)
#STOP added

df = df.sample(frac=1)
arr = df.to_numpy()
splitarr = np.array_split(arr, 5, axis=0)
train = np.concatenate((splitarr[0], splitarr[1], splitarr[3]), axis=0)
test = np.concatenate((splitarr[2], splitarr[4]), axis=0)
a, b = train.shape
e, f = test.shape
Xtrain = train[:,:-1]
ytrain = train[:,b-1].reshape(-1,1)
Xtest = test[:,:-1]
ytest = test[:,f-1]

print(Xtrain.shape, ytrain.shape)
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW", 'y']
df2 = np.concatenate((Xtrain, ytrain), axis=1)
df2 = pd.DataFrame(df2, columns = col_names)
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW"]
df3 = pd.DataFrame(Xtest, columns = col_names)
print(df3)

print(df2)
reg = smf.glm(formula="y ~ P * QRS * T * PQRS * QRST * freqFW", data=df2).fit()
print(reg.summary())

P = df3['P']
QRS = df3['QRS']
T = df3['T']
PQRS = df3['PQRS']
QRST = df3['QRST']
freqFW = df3['freqFW']

z = 0.9208+P	*	2.54E+05+QRS	*	1.55E+07+P*QRS	*	-2.64E+08+T	*	2.12E+06+P*T	*	-5.75E+07+PQRS	*	-4.12E+06+QRST	*	-1.15E+07+P*QRST	*	2.66E+08
z =+QRS*PQRS*QRST	*	-61.6473+P*T*PQRS*QRST	*	-4.72E+05+P*freqFW	*	-8.27E+07+QRS*freqFW	*	-5.15E+09+P*QRS*freqFW	*	5.83E+10+T*freqFW	*	-7.10E+08
z =+P*T*freqFW	*	1.13E+10+P*QRS*T*freqFW	*	5.07E+06+PQRS*freqFW	*	1.34E+09+P*PQRS*freqFW	*	2.22E+06+QRS*PQRS*freqFW	*	2.32E+10+P*QRS*PQRS*freqFW	*	-3.61E+11
z =+P*T*PQRS*freqFW	*	-6.73E+10+QRST*freqFW	*	3.86E+09+P*QRST*freqFW	*	-5.85E+10+P*QRS*QRST*freqFW	*	1.33E+07+P*T*QRST*freqFW	*	-5.25E+09
z =+QRS*T*QRST*freqFW	*	-8.45E+10+P*QRS*T*QRST*freqFW	*	-4.69E+07+PQRS*QRST*freqFW	*	-3.15E+10+P*PQRS*QRST*freqFW	*	3.66E+11+QRS*PQRS*QRST*freqFW	*	9340.1482
z =+P*QRS*PQRS*QRST*freqFW	*	-2.48E+06+T*PQRS*QRST*freqFW	*	8.52E+10+P*T*PQRS*QRST*freqFW	*	2.22E+07+QRS*T*PQRS*QRST*freqFW	*	-1.34E+06+P*QRS*T*PQRS*QRST*freqFW	*	1.26E+07

print(z, np.exp(-z))

expit_z = round(1/(np.exp(-z)+1))
print(expit_z)

import collections
elements_count = collections.Counter(expit_z)
for key, value in elements_count.items():
   print(f"expit_z {key}: {value}")
elements_count = collections.Counter(ytest)
for key, value in elements_count.items():
   print(f"ytest {key}: {value}")

print("zmodel:", np.count_nonzero(np.array(expit_z==ytest))*100/len(ytest))
