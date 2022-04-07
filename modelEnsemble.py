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
import pickle



np.random.seed(457)

#setup
#START added
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW", 'y', '']
df = pd.read_csv('data_aggr_4.csv', delimiter=',', header=None, names=col_names)
del df[df.columns[0]]
#del df[df.columns[-1]]
scaler = MaxAbsScaler()
scaler.fit(df)
scaled = scaler.transform(df)
df = pd.DataFrame(scaled, columns=df.columns)
#STOP added

df = df.sample(frac=1)
arr = df.to_numpy()
splitarr = np.array_split(arr, 5, axis=0)
train = np.concatenate((splitarr[0], splitarr[4]), axis=0)
valid = splitarr[1]
test = splitarr[2]
a, b = train.shape
c, d = valid.shape
e, f = test.shape
Xtrain = train[:,:-1]
ytrain = train[:,b-1].reshape(-1,1)
Xvalid = valid[:,:-1]
yvalid = valid[:,d-1]
Xtest = test[:,:-1]
ytest = test[:,f-1]

print(Xtrain.shape, ytrain.shape)
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW", 'y']
df2 = np.concatenate((Xtrain, ytrain), axis=1)
df2 = pd.DataFrame(df2, columns = col_names)
col_names = ['P', 'QRS', 'T', "PQRS", "QRST", "freqFW"]
df3 = pd.DataFrame(Xtest, columns = col_names)
df4 = pd.DataFrame(Xvalid, columns = col_names)
print(df2, df3)
# kscores = []
# for k in range(1,c):
#     cv = KFold(n_splits=5, random_state=1, shuffle=True)
#     knnModel = KNeighborsClassifier(k)
#     scores = cross_val_score(knnModel, Xvalid, yvalid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#     kscores.append(np.mean(np.abs(scores)))
#
# kscores = np.concatenate((np.arange(1,c).reshape(c-1,1), np.array(kscores).reshape(c-1,1)),axis=1)
# print(kscores)
# kscores = np.array([kscores[i,:] for i in range(c-1) if np.isnan(kscores[i,1]) == False])
# print(kscores)
kscore = 3




models = []
accuracies = []
# Training Model 1
nnModel = KNeighborsClassifier(n_neighbors=int(kscore))
models.append(("knnModel", nnModel))
nnModel.fit(Xtrain, ytrain)
ypred = nnModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 2
dtModel = DecisionTreeClassifier()
models.append(("dtModel1", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 3
dtModel = DecisionTreeClassifier()
models.append(("dtModel2", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 4
dtModel = DecisionTreeClassifier()
models.append(("dtModel3", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

print("Model Accuracies", accuracies)

ensemble = VotingClassifier(models)
ec = ensemble.fit(Xtrain, ytrain)

pickle.dump(ec, open ('vfibTach.pkl','wb'))

start = time()
pred = ec.predict(Xtest)
ens_acc = accuracy_score(pred, ytest)
ens_f1 = f1_score(pred, ytest)
print("Ensemble accuracy is: ", ens_acc)
print("Ensemble f-1 score is: ", ens_f1)
print(confusion_matrix(pred, ytest))
print(classification_report(pred, ytest))
print(ytest.shape)

stop = time() - start
print(str(stop) + "s")
