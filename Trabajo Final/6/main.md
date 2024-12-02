### **Descripción del Código**

1. **Carga del conjunto de datos:**
   - Se utiliza el conjunto de datos `digits` de `sklearn.datasets`, que contiene imágenes de dígitos de 8x8 píxeles.
   - Los datos se almacenan como matrices aplanadas, cada una representando una imagen de dígito.

2. **Preprocesamiento:**
   - Se aplica PCA (Análisis de Componentes Principales) para reducir la dimensionalidad de los datos a 2 dimensiones con fines de visualización.

3. **Entrenamiento de \(K\)-Means:**
   - Se configura \(K\)-Means para crear 10 clusters (\(k = 10\)).
   - El modelo agrupa las imágenes en clusters basándose en sus características.

4. **Visualización:**
   - Los datos reducidos a 2D se visualizan para mostrar cómo se distribuyen los clusters.
   - Para cada cluster, se muestran ejemplos representativos en formato de imagen.

---