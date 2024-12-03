import pandas as pd
import numpy as np
import os

def crear_datos_ejemplo():
    """
    Crea archivos CSV de ejemplo para demostración.
    """
    # Datos para la primera tienda
    datos_tienda1 = {
        'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares'],
        'subtotal': [1200, 25, 45, 300, 50],
        'total': [1400, 30, 55, 350, 60]
    }
    
    # Datos para la segunda tienda
    datos_tienda2 = {
        'producto': ['Laptop', 'Impresora', 'Tablet', 'Monitor', 'Mouse'],
        'subtotal': [1100, 200, 350, 280, 20],
        'total': [1300, 240, 420, 330, 25]
    }
    
    # Crear directorio si no existe
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Guardar los datos en archivos CSV
    pd.DataFrame(datos_tienda1).to_csv('data/tienda1.csv', index=False)
    pd.DataFrame(datos_tienda2).to_csv('data/tienda2.csv', index=False)

def analizar_ventas_tienda(archivo):
    """
    Analiza los datos de ventas de una tienda individual.
    
    Args:
        archivo: Ruta al archivo CSV de la tienda
    Returns:
        tuple: (DataFrame de la tienda, diccionario con estadísticas)
    """
    # Leer el archivo CSV
    df = pd.read_csv(archivo)
    
    # Calcular estadísticas
    estadisticas = {
        'nombre_tienda': os.path.basename(archivo).replace('.csv', ''),
        'total_ventas': df['subtotal'].sum(),
        'promedio_ventas': df['subtotal'].mean(),
        'mediana_ventas': df['subtotal'].median(),
        'desviacion_estandar': df['subtotal'].std(),
        'producto_mas_vendido': df.loc[df['total'].idxmax(), 'producto']
    }
    
    return df, estadisticas

def analizar_todas_tiendas(directorio_csvs):
    """
    Analiza los datos de ventas de todas las tiendas.
    """
    # Listas para almacenar DataFrames y estadísticas
    dfs_tiendas = []
    estadisticas_tiendas = []
    
    # Verificar si el directorio existe
    if not os.path.exists(directorio_csvs):
        raise FileNotFoundError(f"El directorio {directorio_csvs} no existe")
    
    # Procesar cada archivo CSV en el directorio
    archivos_csv = [f for f in os.listdir(directorio_csvs) if f.endswith('.csv')]
    
    if not archivos_csv:
        raise FileNotFoundError("No se encontraron archivos CSV en el directorio")
    
    for archivo in archivos_csv:
        ruta_completa = os.path.join(directorio_csvs, archivo)
        df_tienda, stats_tienda = analizar_ventas_tienda(ruta_completa)
        dfs_tiendas.append(df_tienda)
        estadisticas_tiendas.append(stats_tienda)
    
    # Combinar todos los DataFrames
    df_global = pd.concat(dfs_tiendas, ignore_index=True)
    
    # Calcular estadísticas globales
    estadisticas_globales = {
        'total_empresa': df_global['subtotal'].sum(),
        'promedio_empresa': df_global['subtotal'].mean(),
        'mediana_empresa': df_global['subtotal'].median(),
        'desviacion_estandar_empresa': df_global['subtotal'].std(),
        'producto_mas_vendido_empresa': df_global.loc[df_global['total'].idxmax(), 'producto']
    }
    
    return {
        'estadisticas_globales': estadisticas_globales,
        'estadisticas_tiendas': estadisticas_tiendas
    }

def imprimir_resultados(resultados):
    """
    Imprime los resultados del análisis de manera formateada.
    """
    print("\n=== ESTADÍSTICAS GLOBALES DE LA EMPRESA ===")
    print(f"Total de ventas: ${resultados['estadisticas_globales']['total_empresa']:,.2f}")
    print(f"Promedio de ventas: ${resultados['estadisticas_globales']['promedio_empresa']:,.2f}")
    print(f"Mediana de ventas: ${resultados['estadisticas_globales']['mediana_empresa']:,.2f}")
    print(f"Desviación estándar: ${resultados['estadisticas_globales']['desviacion_estandar_empresa']:,.2f}")
    print(f"Producto más vendido: {resultados['estadisticas_globales']['producto_mas_vendido_empresa']}")
    
    print("\n=== ESTADÍSTICAS POR TIENDA ===")
    for tienda in resultados['estadisticas_tiendas']:
        print(f"\nTienda: {tienda['nombre_tienda']}")
        print(f"Total de ventas: ${tienda['total_ventas']:,.2f}")
        print(f"Promedio de ventas: ${tienda['promedio_ventas']:,.2f}")
        print(f"Mediana de ventas: ${tienda['mediana_ventas']:,.2f}")
        print(f"Desviación estándar: ${tienda['desviacion_estandar']:,.2f}")
        print(f"Producto más vendido: {tienda['producto_mas_vendido']}")

if __name__ == "__main__":
    try:
        # Crear datos de ejemplo
        crear_datos_ejemplo()
        
        # Analizar los datos
        resultados = analizar_todas_tiendas('data')
        imprimir_resultados(resultados)
        
    except Exception as e:
        print(f"Error al procesar los datos: {str(e)}")