import collections
import numpy as np
import pandas as pd
from math import sqrt, exp, pi


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
                v[0, j] = v[0, j] + (set_entrenamiento[i, j + 2]) ** 2
        if set_entrenamiento[i, 1] == 2:
            for j in range(0, cantidad_caracteristicas):
                m[1, j] = m[1, j] + set_entrenamiento[i, j + 2]
                v[1, j] = v[1, j] + (set_entrenamiento[i, j + 2]) ** 2
        if set_entrenamiento[i, 1] == 3:
            for j in range(0, cantidad_caracteristicas):
                m[2, j] = m[2, j] + set_entrenamiento[i, j + 2]
                v[2, j] = v[2, j] + (set_entrenamiento[i, j + 2]) ** 2

    for i in range(0, (len(cantidad_clases))):
        for j in range(0, cantidad_caracteristicas):
            m[i, j] = m[i, j] / cantidad_clases[i + 1]
            v[i, j] = (v[i, j] / cantidad_clases[i + 1]) - (m[i, j] ** 2)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Media de cada caracteristica (columna) para cada clase (fila - densidad)'
          '\n---------------------------------\n',
          m)

    print('\n---------------------------------------------------------------------------------------------------\n'
          'Varianza de cada caracteristica (columna) para cada clase (fila - densidad)'
          '\n---------------------------------\n',
          v)

    return (m, v);


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
          'Probabilidad posterior de las caracteristica dada una clase\n'
          '\tdensidad 1\t-\tdensidad 2\t-\tdensidad 3'
          '\n---------------------------------\n',
          pcc)

    return pcc


# CALCULO DE LA PROBABILIDAD CONDICIONAL PARA CADA CLASE USANDO EL TEOREMA DE BAYES
# Funcion para calcular la media y varianza de los datos, teniendo como parametros `x`
# que es un array con los datos de las caracteristicas, `y` que contiene la clase a la
# que pertenece cada set de caracteristicas y `set_test` que son los datos de prueba


def bayes_naive_gussiano(caracteristicas, clases, set_entrenamiento, set_test, clases_set_test):
    # Llamando los metodos que calculan la media, varianza y probabilidades
    m, v = media_varianza(caracteristicas, clases, set_entrenamiento)
    prob_previa = pre_prob(clases)
    pcc = prob_caracteristica_clase(m, v, set_test)
    cant_clases = m.shape[0]
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

    print('\n\naciertos:', acierto, '\tdesaciertos:', desacierto)


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

# caracteristicas contiene los valores de PD1, PD2, PD3 y PD4 para el entrenamiento
caracteristicas = set_entrenamiento[:, [2, 3, 4, 5]]

# clases contiene el valor de la densidad para cada planta para el entrenamiento
clases = set_entrenamiento[:, 1]

bayes_naive_gussiano(caracteristicas, clases, set_entrenamiento, set_test, clases_set_test)
