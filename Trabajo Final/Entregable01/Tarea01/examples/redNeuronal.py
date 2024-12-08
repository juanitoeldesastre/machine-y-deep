import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Declaramos los datos de entrada
entrada = np.array([2, 3, 60, 14, 35, 43, 252, 402, 2010, 50], dtype=float)
resultados = np.array([0.0254, 0.1524, 0.762, 0.1778, 1.778, 1.0922, 12.776, 5.1054, 25.527, 2.514], dtype=float)

# Topografía de la red
capa1 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Creamos el modelo
modelo = tf.keras.Sequential([capa1])

# Asignamos optimizador y métrica de pérdida
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Se está entrenando la red...")

# Entrenamiento del modelo
entrenamiento = modelo.fit(entrada, resultados, epochs=500, verbose=False)

# Guardamos la red
modelo.save('RedNeuronal.h5')
modelo.save_weights('peso.weights.h5')

# Observar el comportamiento de nuestra red

plt.xlabel("Ciclos de entrenamiento")
plt.ylabel("Errores")
plt.plot(entrenamiento.history["loss"])
plt.show()

# Verificamos que la red se entrenó
print("Entrenamiento terminado")

# Predicción
while True:
    try:
        # Solicitar entrada al usuario
        i = float(input("Ingresar el valor en pulgadas: "))
        
        # Hacer la predicción
        prediccion = modelo.predict(np.array([i]))  # Convertir el input en un array de Numpy
        print(f"El valor en metros es: {prediccion[0][0]}")  # Mostrar el primer valor de la predicción

    except ValueError:
        print("Por favor, ingrese un número válido.")
    except KeyboardInterrupt:
        print("\nTerminando el programa.")
        break
