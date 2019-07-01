# Importing necessary libraries...
import collections
import numpy as np
import pandas as pd
from math import sqrt, exp


def pre_prob(y):
    y_dict = collections.Counter(y)
    pre_probab = np.ones(2)
    for i in range(0, 2):
        pre_probab[i] = y_dict[i]/y.shape[0]
    return pre_probab

#Funcion que calcula la media y varianza

def mean_var(X, y):
    print('data inicial: ',X)
    print('vector de coincidencias de clases: ', y)
    n_features = X.shape[1]
    # Se crean los array m y v que son de 2x3 dimensiones para este caso.
    # Van a contener la media y varianza de cada clase
    m = np.ones((2, n_features))
    v = np.ones((2, n_features))
    n_0 = np.bincount(y.astype(int))[np.nonzero(np.bincount(y.astype(int)))[0]][0]
    print('n_0',n_0)
    x0 = np.ones((n_0, n_features))
    x1 = np.ones((X.shape[0] - n_0, n_features))

    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 0:
            x0[k] = X[i]
            k = k + 1
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 1:
            x1[k] = X[i]
            k = k + 1

    for j in range(0, n_features):
        m[0][j] = np.mean(x0.T[j])
        v[0][j] = np.var(x0.T[j]) * (n_0 / (n_0 - 1))
        m[1][j] = np.mean(x1.T[j])
        v[1][j] = np.var(x1.T[j]) * ((X.shape[0] - n_0) / ((X.shape[0] - n_0) - 1))
    return m, v  # mean and variance

def prob_feature_class(m, v, x):
    n_features = m.shape[1]
    pfc = np.ones(2)
    for i in range(0, 2):
        product = 1
        for j in range(0, n_features):
            product = product * (1/sqrt(2*3.14*v[i][j])) * exp(-
                                  pow((x[j] - m[i][j]),2)/v[i][j]*2)
        pfc[i] = product
    print('pfc\n',pfc)
    return pfc

def GNB(X, y, x):
    m, v = mean_var(X, y)
    pfc = prob_feature_class(m, v, x)
    pre_probab = pre_prob(y)
    pcf = np.ones(2)
    total_prob = 0
    for i in range(0, 2):
        total_prob = total_prob + (pfc[i] * pre_probab[i])
    for i in range(0, 2):
        pcf[i] = (pfc[i] * pre_probab[i])/total_prob
    prediction = int(pcf.argmax())
    print('prediction\n\n',pcf)
    return m, v, pre_probab, pfc, pcf, prediction

data = pd.read_csv('./GenderData.csv', delimiter=',')
# print(data)

# converting from pandas to numpy ...
X_train = np.array(data.iloc[:,[1,2,3]])
y_train = np.array(data['Person'])
print(y_train)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

for i in range(0,y_train.shape[0]):
    if y_train[i] == "female":
        y_train[i] = 0
    else:
        y_train[i] = 1

x = np.array([6, 130, 8]) # set de prueba

# executing the Gaussian Naive Bayes for the test instance...
m, v, pre_probab, pfc, pcf, prediction = GNB(X_train, y_train, x)
print('Mean---------------------------------------------------------')
print(m) # Output given below...(mean for 2 classes of all features)
print('Variance---------------------------------------------------------')
print(v) # Output given below..(variance for 2 classes of features)
print('Prior Probabilities---------------------------------------------------------')
print(pre_probab) # Output given below.........(prior probabilities)
print('Posterior probabilities---------------------------------------------------------')
print(pfc) # Output given below............(posterior probabilities)
print('Conditional Probability of the classes given test-data---------------------------------------------------------')
print(pcf) # Conditional Probability of the classes given test-data
print('Prediccion---------------------------------------------------------')
print(prediction) # Output given below............(final prediction)

