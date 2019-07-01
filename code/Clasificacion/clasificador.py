import collections
import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# CALCULO DE MEDIAS Y VARIANZAS
# Funcion para calcular la media y varianza de los datos, teniendo como parametros `x` que es un array con los datos
# de las caracteristicas y `y` que contiene la clase a la que pertenece cada set de caracteristicas
def media_varianza(x, y, set_entrenamiento):
    cantidad_caracteristicas = x.shape[1]
    # Se crean los array m y v que son de 2x3 dimensiones para este caso.
    # Van a contener la media y varianza de cada clase, las columnas tienen el mismo orden de las caracteristicas, y las
    # filas son el resultado de la mediana o varianza de cada caracteristica para esa clase
    cantidad_clases = collections.Counter(y)
    m = np.zeros((len(cantidad_clases), cantidad_caracteristicas))
    v = np.zeros((len(cantidad_clases), cantidad_caracteristicas))

    # El siguiente ciclo recorre la data y va sumando los valores por
    # caracteristica para esa fila de datos, con esto se calculara la media
    for i in range(0, len(set_entrenamiento)):
        if set_entrenamiento[i, 1] == 1:
            for j in range(0, cantidad_caracteristicas):
                m[0, j] = m[0, j] + set_entrenamiento[i, j + 2]
        if set_entrenamiento[i, 1] == 2:
            for j in range(0, cantidad_caracteristicas):
                m[1, j] = m[1, j] + set_entrenamiento[i, j + 2]
        if set_entrenamiento[i, 1] == 3:
            for j in range(0, cantidad_caracteristicas):
                m[2, j] = m[2, j] + set_entrenamiento[i, j + 2]

    for i in range(0, (len(cantidad_clases))):
        for j in range(0, cantidad_caracteristicas):
            m[i, j] = m[i, j] / cantidad_clases[i + 1]

    for i in range(0, len(set_entrenamiento)):
        if set_entrenamiento[i, 1] == 1:
            for j in range(0, cantidad_caracteristicas):
                v[0, j] = v[0, j] + ((set_entrenamiento[i, j + 2] - m[0, j]) ** 2)
        if set_entrenamiento[i, 1] == 2:
            for j in range(0, cantidad_caracteristicas):
                v[1, j] = v[1, j] + ((set_entrenamiento[i, j + 2] - m[1, j]) ** 2)
        if set_entrenamiento[i, 1] == 3:
            for j in range(0, cantidad_caracteristicas):
                v[2, j] = v[2, j] + ((set_entrenamiento[i, j + 2] - m[2, j]) ** 2)

    for i in range(0, (len(cantidad_clases))):
        for j in range(0, cantidad_caracteristicas):
            v[i, j] = v[i, j] / (cantidad_clases[i + 1]-1)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Media de cada caracteristica (columna) para cada clase (fila - densidad)'
          '\n---------------------------------\n',
          m)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Varianza de cada caracteristica (columna) para cada clase (fila - densidad)'
          '\n---------------------------------\n',
          v)

    return (m, v)


# CALCULO DE PROBABILIDADES PREVIAS
# Funcion para calcular la probabilidad previa de cada clase
def pre_prob(y):
    cantidad_conteo_clases = collections.Counter(y)
    prob = np.ones(len(cantidad_conteo_clases))
    for i in range(0, len(cantidad_conteo_clases)):
        prob[i] = cantidad_conteo_clases[i + 1] / len(y)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Probabilidad Previa para cada clase (columna - densidad)\n'
          'densidad 1 - densidad 2 - densidad 3'
          '\n---------------------------------\n',
          prob)
    return prob


# CALCULO DE PROBABILIDAD POSTERIOR DE CADA CARACTERISTICA DEL SET DE PRUEBA POR CLASE
# Funcion que calcula la probabilidad posterior del set de datos de prueba
# dada la clase c

def prob_caracteristica_clase(m, v, set_test):
    cant_caracteristicas = m.shape[1]
    cant_clases = m.shape[0]
    pcc = np.zeros((len(set_test), cant_clases))

    for i in range(0, len(set_test)):
        for j in range(0, cant_clases):
            producto = 1
            for k in range(0, cant_caracteristicas):
                producto = producto * (1 / sqrt(2 * pi * v[j][k])) * exp(
                    -(set_test[i][k] - m[j][k]) ** 2 / (2 * v[j][k]))
            pcc[i][j] = producto

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Probabilidad posterior de las caracteristicas dada una clase\n'
          '\tdensidad 1\t-\tdensidad 2\t-\tdensidad 3'
          '\n---------------------------------\n',
          pcc)

    return pcc



# CALCULO DE LA PROBABILIDAD CONDICIONAL PARA CADA CLASE USANDO EL TEOREMA DE BAYES
# Funcion para calcular la media y varianza de los datos, teniendo como parametros `x`
# que es un array con los datos de las caracteristicas, `y` que contiene la clase a la
# que pertenece cada set de caracteristicas y `set_test` que son los datos de prueba
def prob_condicional(set_test, cant_clases, pcc, prob_previa, clases_set_test):
    prob_cond = np.zeros((len(set_test), cant_clases))
    prob_total = np.zeros(len(set_test))

    for i in range(0, len(set_test)):
        for j in range(0, cant_clases):
            prob_total[i] = prob_total[i] + (pcc[i][j] * prob_previa[j])

    for i in range(0, len(set_test)):
        for j in range(0, cant_clases):
            prob_cond[i][j] = (pcc[i][j] * prob_previa[j]) / prob_total[i]

    prediccion = np.zeros(len(set_test))
    acierto = 0
    desacierto = 0

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Probabilidad condicional para cada clase\n'
          'densidad 1\t-\tdensidad 2\t-\tdensidad 3'
          '\n---------------------------------\n',
          prob_cond)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Prediccion \t '
          'Valores reales\n---------------------------------\n')
    for i in range(0, len(set_test)):
        prediccion[i] = (prob_cond[i].argmax() + 1)
        print(prediccion[i], '\t\t\t', clases_set_test[i])
        if (prediccion[i] == clases_set_test[i]):
            acierto += 1
        else:
            desacierto += 1

    print('\n\naciertos:', acierto / (acierto + desacierto), '\tdesaciertos:', desacierto / (acierto + desacierto))
    return prob_cond, prediccion


# Curvas ROC
def curvas_roc(clases_set_test, prediccion, cant_clases, prob_cond):
    # https://stackoverflow.com/questions/50941223/plotting-roc-curve-with-multiple-classes
    # http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/
    matrix_confusion = confusion_matrix(clases_set_test, prediccion)
    print('\n---------------------------------------------------------------------------------------------------\n'
          'Matriz de confusión'
          '\n---------------------------------\n',
          matrix_confusion)
    # plt.style.use('ggplot')
    # Compute ROC curve and ROC AUC for each class

    y_test = np.zeros((len(clases_set_test), cant_clases))
    for i in range(0, len(clases_set_test)):
        if (clases_set_test[i] == 1):
            y_test[i][0] = 1
        if (clases_set_test[i] == 2):
            y_test[i][1] = 1
        if (clases_set_test[i] == 3):
            y_test[i][2] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(y_test)
    print()
    for i in range(cant_clases):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prob_cond[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = (['blue', 'red', 'green'])
    for i, color in zip(range(cant_clases), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='Curva ROC de la clase {0} (área = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva característica operativa del receptor para datos multiclase')
    plt.legend(loc="lower right")
    plt.show()


def bayes_naive_gaussiano(set_entrenamiento, set_test, clases_set_test):
    # caracteristicas contiene los valores de PD1, PD2, PD3 y PD4 para el entrenamiento
    caracteristicas = set_entrenamiento[:, [2, 3, 4, 5]]
    # clases contiene el valor de la densidad para cada planta para el entrenamiento
    clases = set_entrenamiento[:, 1]
    # Llamando los metodos que calculan la media, varianza y probabilidades
    m, v = media_varianza(caracteristicas, clases, set_entrenamiento)
    prob_previa = pre_prob(clases)
    pcc = prob_caracteristica_clase(m, v, set_test)
    cant_clases = m.shape[0]
    prob_cond, prediccion = prob_condicional(set_test, cant_clases, pcc, prob_previa, clases_set_test)
    curvas_roc(clases_set_test, prediccion, cant_clases, prob_cond)


# LECTURA DEL ARCHIVO Y CREACION DE SETS DE ENTRENAMIENTO Y TEST
# Leyendo la data de un archivo .csv
data = pd.read_csv('./datag.csv', delimiter=',')

# pasando la data de pandas dataframe a numpy ndarray
data = np.array(data.iloc[:])

# La data esta ordenada segun densidad, se le realiza un shuffle para desordenarla
np.random.shuffle(data)

# determinando los sets de prueba y entrenamiento
# primero se calcular la cantidad de elementos que va a tener el set de pruebas, en este caso el valor va
# a ser el 20% de los datos

cantidad_muestras = int(len(data) * 0.2)

# se crean dos array a partir de data que van a ser el de test y entrenamiento

set_test = data[0:cantidad_muestras, [2, 3, 4, 5]]
clases_set_test = data[0:cantidad_muestras, 1]
set_entrenamiento = data[cantidad_muestras: len(data), :]

bayes_naive_gaussiano(set_entrenamiento, set_test, clases_set_test)

