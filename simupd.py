import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class ClimateModel:
    def __init__(self, initial_temperature, initial_precipitation):
        self.temperature = [initial_temperature]  # Utilizamos el valor inicial proporcionado
        self.precipitation_pattern = [initial_precipitation]  # Utilizamos el valor inicial proporcionado

    def predict_next_10_years(self, data):
        X = data[['Year']].values
        y_temp = data['Temperature'].values
        y_precip = data['Precipitation'].values

        # Ajustar modelos de regresión polinomial
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)

        lin_reg_temp = LinearRegression()
        lin_reg_temp.fit(X_poly, y_temp)

        lin_reg_precip = LinearRegression()
        lin_reg_precip.fit(X_poly, y_precip)

        # Predecir los próximos 10 años
        for i in range(1, 11):
            next_year = data['Year'].iloc[-1] + i
            next_year_poly = poly_features.transform([[next_year]])
            next_temp = lin_reg_temp.predict(next_year_poly)[0]
            next_precip = lin_reg_precip.predict(next_year_poly)[0]
            self.temperature.append(next_temp)
            self.precipitation_pattern.append(next_precip)

    def plot_temperature(self):
        plt.plot(range(len(self.temperature)), self.temperature, label='Simulated Temperature')
        plt.xlabel('Years')
        plt.ylabel('Temperature (°C)')
        plt.title('Climate Change Simulation - Temperature')
        plt.legend()
        plt.show()

    def plot_precipitation(self):
        plt.plot(range(len(self.precipitation_pattern)), self.precipitation_pattern, label='Simulated Precipitation')
        plt.xlabel('Years')
        plt.ylabel('Precipitation Pattern')
        plt.title('Climate Change Simulation - Precipitation')
        plt.legend()
        plt.show()

# Carga de datos CSV con Pandas
data = pd.read_csv('datosclim.csv')

# Obtener el primer valor de temperatura y precipitación del archivo CSV como valores iniciales
initial_temperature = data['Temperature'].iloc[0]
initial_precipitation = data['Precipitation'].iloc[0]

# Crear instancia del modelo climático
model = ClimateModel(initial_temperature, initial_precipitation)

# Predecir los próximos 10 años utilizando los datos cargados
model.predict_next_10_years(data)

# Graficar la temperatura simulada
model.plot_temperature()

# Graficar el patrón de precipitación simulado
model.plot_precipitation()
