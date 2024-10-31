import os
import subprocess

import numpy as np
import tensorflow as tf

# Directorio base donde guardaremos los conjuntos
base_save_path = os.path.join("..", "..", "data", "processed")

val_save_path = os.path.join(base_save_path, "val")
test_save_path = os.path.join(base_save_path, "test")

# Cargar el dataset guardado
val = tf.data.Dataset.load(val_save_path)
test = tf.data.Dataset.load(test_save_path)
print("Dataset cargado con éxito.")

import glob

# Directorio donde guardas los modelos
models_dir = os.path.join("..", "..", "src", "models")

# Buscar todos los archivos de modelo en el directorio que terminan en .keras
model_files = glob.glob(os.path.join(models_dir, "saved_model_*.keras"))

# Seleccionar el archivo de modelo más reciente
latest_model_file = max(model_files, key=os.path.getctime)

# Cargar el modelo más reciente
model = tf.keras.models.load_model(latest_model_file)
print(f"Modelo más reciente cargado: {latest_model_file}")


y_true = []
y_pred = []

for images, labels in test:  # Utilizando test en lugar de imágenes individuales
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))  # Etiquetas verdaderas
    y_pred.extend(np.argmax(predictions, axis=1))  # Predicciones

y_true = np.array(y_true)
y_pred = np.array(y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Crear matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión")
plt.show()
