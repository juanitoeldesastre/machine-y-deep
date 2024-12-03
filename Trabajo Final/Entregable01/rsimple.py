import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Variable independiente
y = np.array([1.5, 3.7, 2.8, 4.9, 6.2])       # Variable dependiente

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X, y)

# Predicciones
y_pred = model.predict(X)

# Visualización de los resultados
plt.scatter(X, y, color='blue', label='Datos originales')
plt.plot(X, y_pred, color='red', label='Línea de regresión')
plt.title('Regresión Lineal Simple')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Coeficientes del modelo
print("Pendiente (β1):", model.coef_[0])
print("Intercepto (β0):", model.intercept_)
