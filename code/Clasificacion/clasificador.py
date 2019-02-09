import collections
import numpy as np
import pandas as pd
from math import sqrt, exp


# Funcion para calcular la media y varianza de los datos, teniendo como parametros `x` que es un array con los datos
# de los atributos y `y` que contiene la clase a la que pertenece cada set de atributos
def media_varianza (x, y, set_entrenamiento):
    cantidad_atributos = x.shape[1]
    # Se crean los array m y v que son de 2x3 dimensiones para este caso.
    # Van a contener la media y varianza de cada clase, las columnas tienen el mismo orden de los atributos, y las
    # filas son el resultado de la mediana o varianza de cada atributo para esa clase
    cantidad_clases = collections.Counter(y)
    m = np.zeros((len(cantidad_clases), cantidad_atributos))
    v = np.zeros((len(cantidad_clases), cantidad_atributos))
    conteo_clases = np.zeros((len(cantidad_clases)))

# El siguiente ciclo recorre la data y cuanta la cantidad de repeticiones de cada clase y va sumando los valores por
# atributo para esa fila de datos
    for i in range(0 , len(set_entrenamiento)):
        if set_entrenamiento[i , 1] == 1:
            conteo_clases[0] = conteo_clases[0] + 1
            for j in range(0, cantidad_atributos):
                m[0,j] = m[0,j] + set_entrenamiento[i , j+2]
        if set_entrenamiento[i , 1] == 2:
            conteo_clases[1] = conteo_clases[1] + 1
            for j in range(0, cantidad_atributos):
                m[1,j] = m[1,j] + set_entrenamiento[i , j+2]
        if set_entrenamiento[i, 1] == 3:
            conteo_clases[2] = conteo_clases[2] + 1
            for j in range(0, cantidad_atributos):
                m[2,j] = m[2,j] + set_entrenamiento[i , j+2]

    for i in range(0, (len(cantidad_clases))):
        for j in range(0, cantidad_atributos):
            m[i,j]=m[i,j]/conteo_clases[i]

    print(conteo_clases)
    print(m)


# Leyendo la data de un archivo .csv
data = pd.read_csv('./datag.csv', delimiter=',')

#pasando la data de pandas dataframe a numpy ndarray
data = np.array(data.iloc[:])

# La data esta ordenada segun densidad, se le realiza un shuffle para desordenarla
np.random.shuffle(data)

# determinando los sets de prueba y entrenamienot
# primero se calcular la cantidad de elementos que va a tener el set de muestra, en este caso el valor va
# a ser el 30% de los datos

cantidad_muestras = int(len(data) * 0.3)

# se crean dos array a partir de data que van a ser el de test y entrenamiento

set_test = data[0:cantidad_muestras, :]
set_entrenamiento = data[cantidad_muestras : len(data), :]

# atributos contiene los valores de PD1, PD2, PD3 y PD4 para el entrenamiento
atributos = set_entrenamiento[:,[2,3,4,5]]

# clases contiene el valor de la densidad para cada planta para el entrenamiento
clases = set_entrenamiento[:,1]

media_varianza(atributos, clases, set_entrenamiento)






