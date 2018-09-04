# Regresión lineal usando el dataset de presión sanguínea vs. edad y usando la
# librería Keras.
# Tomado de: http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html
# 
# codificandobits.com - 2018

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

#
# Lectura y visualización del set de datos
#

datos = pd.read_csv('dataset.csv', sep=",", skiprows=32, usecols=[2,3])
print(datos)

# Al graficar los datos se observa una tendencia lineal
datos.plot.scatter(x='Age', y='Systolic blood pressure')
plt.xlabel('Edad (años)')
plt.ylabel('Presión sistólica (mm de Mercurio)')
plt.show()

x = datos['Age'].values
y = datos['Systolic blood pressure'].values

#
# Construir el modelo en Keras
#

# - Capa de entrada: 1 dato (cada dato "x" correspondiente a la edad)
# - Capa de salida: 1 dato (cada dato "y" correspondiente a la regresión lineal)
# - Activación: 'linear' (pues se está implementando la regresión lineal)

np.random.seed(2)			# Para reproducibilidad del entrenamiento

input_dim = 1
output_dim = 1
modelo = Sequential()
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# Definición del método de optimización (gradiente descendiente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático
# medio

sgd = SGD(lr=0.0004)
modelo.compile(loss='mse', optimizer=sgd)

# Imprimir en pantalla la información del modelo
modelo.summary()

#
# Entrenamiento: realizar la regresión lineal
#

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
# iteración (batch_size = 29)

num_epochs = 40000
batch_size = x.shape[0]
history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

#
# Visualizar resultados del entrenamiento
#

# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Graficar el error vs epochs y el resultado de la regresión
# superpuesto a los datos originales
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')

y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datos originales y regresión lineal')
plt.show()

# Predicción
x_pred = np.array([90])
y_pred = modelo.predict(x_pred)
print("La presión sanguínea será de {:.1f} mm-Hg".format(y_pred[0][0]), " para una persona de {} años".format(x_pred[0]))
