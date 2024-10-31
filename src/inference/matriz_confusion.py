import os

import numpy as np
import tensorflow as tf

# Directorio base donde guardaremos los conjuntos
base_load_path = os.path.join("..", "..", "data", "processed")

val_load_path = os.path.join(base_load_path, "val")
test_load_path = os.path.join(base_load_path, "test")

# Cargo el dataset guardado
val = tf.data.Dataset.load(val_load_path)
test = tf.data.Dataset.load(test_load_path)
print("Dataset cargado con éxito.")

import glob

# Directorio donde guardo los modelos
models_dir = os.path.join("..", "..", "src", "models")

# Busco todos los archivos .keras
model_files = glob.glob(os.path.join(models_dir, "saved_model_*.keras"))

# Selecciono el archivo de modelo más reciente
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

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión")

# Guardar ----------------------------------------------------------------
# Crea la ruta y asegura de que exista
save_dir = os.path.join("..", "..", "reports", "figures")
os.makedirs(save_dir, exist_ok=True)  # Crea la carpeta si no existe

# Guardar la figura en la ruta especificada
save_path = os.path.join(save_dir, "matriz_confusion.jpg")
plt.savefig(save_path)  # Ajustar dpi si necesitas mayor resolución

print(f"Figura guardada en {save_path}")
