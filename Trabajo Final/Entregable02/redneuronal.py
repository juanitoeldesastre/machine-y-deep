import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar y preprocesar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos a valores entre 0 y 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir las etiquetas a formato one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Definir el modelo de red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar las imágenes de 28x28 píxeles
    Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y ReLU
    Dense(64, activation='relu'),   # Otra capa oculta con 64 neuronas
    Dense(10, activation='softmax') # Capa de salida para clasificación en 10 clases
])

# 3. Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# 5. Evaluar el modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}")

# 6. Visualizar el entrenamiento
plt.figure(figsize=(12, 4))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

# 7. Predicción en ejemplos del conjunto de prueba
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Predicción: {predictions[i].argmax()}, Etiqueta Real: {y_test[i].argmax()}')
    plt.axis('off')
    plt.show()
