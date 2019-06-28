import xlrd
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly import tools
import matplotlib.pyplot as plt

plotly.tools.set_credentials_file(username='Estefanycss', api_key='HNgIr2Ezlb21nsKgHvng')

# Read data file
df = pd.read_csv('./datag.csv', delimiter=',')
format = ['densidad', 'PD1', 'PD2', 'PD3', 'PD4', 'x', 'y']
values = df[format]
formatW = ['PD1', 'PD2', 'PD3', 'PD4']
weights = df[formatW]
formatC = ['x', 'y']
coordinates = df[formatC]
density = df['densidad']

# Groups for each density
# [PD1,PD2,PD3,PD4]
filtro_densidad_1 = values.loc[values['densidad'] == 1]
filtro_densidad_2 = values.loc[values['densidad'] == 2]
filtro_densidad_3 = values.loc[values['densidad'] == 3]

# Each weight for each density without 0

p1d1 = filtro_densidad_1.loc[filtro_densidad_1['PD1'] != 0]["PD1"]
p1d2 = filtro_densidad_2.loc[filtro_densidad_2['PD1'] != 0]["PD1"]
p1d3 = filtro_densidad_3.loc[filtro_densidad_3['PD1'] != 0]["PD1"]
p2d1 = filtro_densidad_1.loc[filtro_densidad_1['PD2'] != 0]["PD2"]
p2d2 = filtro_densidad_2.loc[filtro_densidad_2['PD2'] != 0]["PD2"]
p2d3 = filtro_densidad_3.loc[filtro_densidad_3['PD2'] != 0]["PD2"]
p3d1 = filtro_densidad_1.loc[filtro_densidad_1['PD3'] != 0]["PD3"]
p3d2 = filtro_densidad_2.loc[filtro_densidad_2['PD3'] != 0]["PD3"]
p3d3 = filtro_densidad_3.loc[filtro_densidad_3['PD3'] != 0]["PD3"]
p4d1 = filtro_densidad_1.loc[filtro_densidad_1['PD4'] != 0]["PD4"]
p4d2 = filtro_densidad_2.loc[filtro_densidad_2['PD4'] != 0]["PD4"]
p4d3 = filtro_densidad_3.loc[filtro_densidad_3['PD4'] != 0]["PD4"]

# Descriptive statistics
print("------------- Estadisticas descriptivas para todos los valores")
print(values.describe())
print(density.describe())
print(weights.describe())

# Mediana
print("------------- Medianas")
print(values.median())

# Varianza
print("------------- Varianzas")
print(density.var())
print(weights.var())

# Moda
print("------------- Moda")
print(density.mode())
print(weights.mode())

# data = [go.Histogram(x=values['density'])]
# plotly.plot(data, filename='pandas/simple-histogram')

grouped_data = values.groupby(['densidad'])
print("------------- Estadisticas descriptivas agrupando por densidad")
print("Peso fresco calibre 1")
print(grouped_data['PD1'].describe())
print("Peso fresco calibre 2")
print(grouped_data['PD2'].describe())
print("Peso fresco calibre 3")
print(grouped_data['PD3'].describe())
print("Peso fresco calibre 4")
print(grouped_data['PD4'].describe())

# Histograms (weights for each density)

trace1 = go.Histogram(x=p1d1, opacity=0.3, name="Peso calibre 1 para densidad 1")
trace2 = go.Histogram(x=p1d2, opacity=0.3, name="Peso calibre 1 para densidad 2")
trace3 = go.Histogram(x=p1d3, opacity=0.3, name="Peso calibre 1 para densidad 3")
trace4 = go.Histogram(x=p2d1, opacity=0.3, name="Peso calibre 2 para densidad 1")
trace5 = go.Histogram(x=p2d2, opacity=0.3, name="Peso calibre 2 para densidad 2")
trace6 = go.Histogram(x=p2d3, opacity=0.3, name="Peso calibre 2 para densidad 3")
trace7 = go.Histogram(x=p3d1, opacity=0.3, name="Peso calibre 3 para densidad 1")
trace8 = go.Histogram(x=p3d2, opacity=0.3, name="Peso calibre 3 para densidad 2")
trace9 = go.Histogram(x=p3d3, opacity=0.3, name="Peso calibre 3 para densidad 3")
trace10 = go.Histogram(x=p4d1, opacity=0.3, name="Peso calibre 4 para densidad 1")
trace11 = go.Histogram(x=p4d2, opacity=0.3, name="Peso calibre 4 para densidad 2")
trace12 = go.Histogram(x=p4d3, opacity=0.3, name="Peso calibre 4 para densidad 3")

data1 = [trace1, trace2, trace3]
layout1 = go.Layout(barmode='overlay', title="Pesos calibre 1 para cada densidad", xaxis=dict(title='Peso (gr)'))
fig1 = go.Figure(data=data1, layout=layout1)
plotly.offline.plot(fig1, filename="pesos1.html")

data2 = [trace4, trace5, trace6]
layout2 = go.Layout(barmode='overlay', title="Pesos calibre 2 para cada densidad", xaxis=dict(title='Peso (gr)'))
fig2 = go.Figure(data=data2, layout=layout2)
plotly.offline.plot(fig2, filename="pesos2.html")

data3 = [trace7, trace8, trace9]
layout3 = go.Layout(barmode='overlay', title="Pesos calibre 3 para cada densidad", xaxis=dict(title='Peso (gr)'))
fig3 = go.Figure(data=data3, layout=layout3)
plotly.offline.plot(fig3, filename="pesos3.html")

data4 = [trace10, trace11, trace12]
layout4 = go.Layout(barmode='overlay', title="Pesos calibre 4 para cada densidad", xaxis=dict(title='Peso (gr)'))
fig4 = go.Figure(data=data4, layout=layout4)
plotly.offline.plot(fig4, filename="pesos4.html")

# Correlation
data_corr_1 = filtro_densidad_1.drop(['x', 'y', 'densidad'], axis=1)
data_corr_2 = filtro_densidad_2.drop(['x', 'y', 'densidad'], axis=1)
data_corr_3 = filtro_densidad_3.drop(['x', 'y', 'densidad'], axis=1)

print("\n------------- Matriz de Correlación para la densidad 1")
print(data_corr_1.corr(method='spearman'))
plt.matshow(data_corr_1.corr(method='spearman'))
plt.title("Matriz de Correlación para la densidad 1", pad=25)
plt.xticks(range(len(data_corr_1.columns)), data_corr_1.columns)
plt.yticks(range(len(data_corr_1.columns)), data_corr_1.columns)
plt.colorbar()
plt.show()

print("\n------------- Matriz de Correlación para la densidad 2")
print(data_corr_2.corr(method='spearman'))
plt.matshow(data_corr_2.corr(method='spearman'))
plt.title("Matriz de Correlación para la densidad 2", pad=25)
plt.xticks(range(len(data_corr_2.columns)), data_corr_2.columns)
plt.yticks(range(len(data_corr_2.columns)), data_corr_2.columns)
plt.colorbar()
plt.show()

print("\n------------- Matriz de Correlación para la densidad 3")
print(data_corr_3.corr(method='spearman'))
plt.matshow(data_corr_3.corr(method='spearman'))
plt.title("Matriz de Correlación para la densidad 3", pad=25)
plt.xticks(range(len(data_corr_3.columns)), data_corr_3.columns)
plt.yticks(range(len(data_corr_3.columns)), data_corr_3.columns)
plt.colorbar()
plt.show()
