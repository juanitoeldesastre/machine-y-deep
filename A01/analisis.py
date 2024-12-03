import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def crear_datos_entrenamiento():
    """
    Crea un conjunto de datos más extenso para entrenamiento del modelo
    """
    productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares', 'Tablet', 'Impresora']
    
    # Crear características para cada producto
    data = []
    np.random.seed(42)
    
    for _ in range(1000):  # Crear 1000 registros
        producto = np.random.choice(productos)
        
        # Características que pueden influir en las ventas
        precio = np.random.normal({
            'Laptop': 1200,
            'Mouse': 25,
            'Teclado': 45,
            'Monitor': 300,
            'Auriculares': 50,
            'Tablet': 350,
            'Impresora': 200
        }[producto], scale=50)
        
        mes = np.random.randint(1, 13)
        dia_semana = np.random.randint(1, 8)
        
        # Variable objetivo: 1 si se vendió, 0 si no
        vendido = np.random.binomial(1, {
            'Laptop': 0.7,
            'Mouse': 0.8,
            'Teclado': 0.75,
            'Monitor': 0.6,
            'Auriculares': 0.65,
            'Tablet': 0.55,
            'Impresora': 0.5
        }[producto])
        
        data.append([producto, precio, mes, dia_semana, vendido])
    
    return pd.DataFrame(data, columns=['producto', 'precio', 'mes', 'dia_semana', 'vendido'])

def preparar_datos(df):
    """
    Prepara los datos para el modelo de machine learning
    """
    # Codificar la variable de producto
    le = LabelEncoder()
    df['producto_encoded'] = le.fit_transform(df['producto'])
    
    # Normalizar características numéricas
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['producto_encoded', 'precio', 'mes', 'dia_semana']])
    y = df['vendido']
    
    return X, y, le

def entrenar_modelo(X, y):
    """
    Entrena el modelo de regresión logística
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Evaluar el modelo
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)
    
    return modelo, X_test, y_test, predicciones, probabilidades

def visualizar_resultados(modelo, X_test, y_test, predicciones, le, df):
    """
    Visualiza los resultados del modelo
    """
    # Crear figura con subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Matriz de confusión
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, predicciones)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    
    # 2. Probabilidades de venta por producto
    plt.subplot(2, 2, 2)
    productos_prob = []
    for producto in le.classes_:
        producto_encoded = le.transform([producto])[0]
        precio_medio = df[df['producto'] == producto]['precio'].mean()
        X_pred = np.array([[producto_encoded, precio_medio, 6, 3]])  # mes=6, dia=3 como ejemplo
        prob = modelo.predict_proba(X_pred)[0][1]
        productos_prob.append((producto, prob))
    
    productos_prob.sort(key=lambda x: x[1], reverse=True)
    productos, probs = zip(*productos_prob)
    
    plt.barh(productos, probs)
    plt.title('Probabilidad de Venta por Producto')
    plt.xlabel('Probabilidad')
    
    # 3. Importancia de características
    plt.subplot(2, 2, 3)
    caracteristicas = ['Producto', 'Precio', 'Mes', 'Día']
    importancia = abs(modelo.coef_[0])
    plt.bar(caracteristicas, importancia)
    plt.title('Importancia de Características')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo.png')
    plt.close()

def main():
    # Crear y preparar datos
    print("Creando conjunto de datos de entrenamiento...")
    df = crear_datos_entrenamiento()
    
    print("\nPreparando datos para el modelo...")
    X, y, le = preparar_datos(df)
    
    print("\nEntrenando modelo...")
    modelo, X_test, y_test, predicciones, probabilidades = entrenar_modelo(X, y)
    
    print("\nVisualizando resultados...")
    visualizar_resultados(modelo, X_test, y_test, predicciones, le, df)
    
    # Imprimir reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, predicciones))
    
    # Mostrar probabilidades de venta por producto
    print("\nProbabilidad de venta por producto:")
    productos_prob = []
    for producto in le.classes_:
        producto_encoded = le.transform([producto])[0]
        precio_medio = df[df['producto'] == producto]['precio'].mean()
        X_pred = np.array([[producto_encoded, precio_medio, 6, 3]])  # mes=6, dia=3 como ejemplo
        prob = modelo.predict_proba(X_pred)[0][1]
        productos_prob.append((producto, prob))
    
    productos_prob.sort(key=lambda x: x[1], reverse=True)
    for producto, prob in productos_prob:
        print(f"{producto}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()