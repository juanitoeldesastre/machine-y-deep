#3. IMPLEMENTAR LOS SIGUIENTES PROGRAMAS EN PYTHON USANDO CON PANDAS:

#1. DataFrame  : crear una estructura diccionario..y..cargar a una dataframe y luego mostrarla

import pandas as pd
#Crear un DataFrame
data = {'Nombre': ['Aria','Luis','Juan'],'Edad':[28,34,29],'Ciudad':['Lima','Cusco','Arequipa']}
df = pd.DataFrame(data)
print(df)

#2. Series : crear una serie de 5 numeros y mostrar en pantalla
#Crear una Serie
serie = pd.Series([1,2,3,4,5], name="Numeros")
print(serie)

#3. Manipulacion de datos : filtrar los que tienen edad mayor a 30 y agregar una nueva columna pais con valor peru
# Guardar el DataFrame en un archivo CSV
df.to_csv("mi_dataframe.csv", index=False)
# Cargar el DataFrame desde un archivo CSV
df_cargado = pd.read_csv("mi_dataframe.csv")
print(df_cargado)

#5. Operaiones estadisticas del dataframe anterior df , sacar la media, maxima y minima de la columna edad
# Calcular estadisticas descriptivas
print(df['Edad'].mean()) # Media
print(df['Edad'].max()) # Maxima
print(df['Edad'].min()) # Minima

#6. Visualizacion de datos : visualizar con grafico de barra variable edad
import matplotlib.pyplot as plt
# Graficar datos
df['Edad'].plot(kind='bar')
plt.xlabel("Indice")
plt.ylabel("Edad")
plt.title("Edades de personas")
plt.show()