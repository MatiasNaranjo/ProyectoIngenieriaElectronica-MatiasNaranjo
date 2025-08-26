import os
import subprocess

import tensorflow as tf

# Ejecutamos el script data.py
# subprocess.run(["python", "../src/data/data.py"])}

# Directorio base
base_save_path = os.path.join("..", "..", "data", "processed")

# Directorio de cada conjunto
train_save_path = os.path.join(base_save_path, "train")
val_save_path = os.path.join(base_save_path, "val")
test_save_path = os.path.join(base_save_path, "test")

# Cargar el dataset guardado
train = tf.data.Dataset.load(train_save_path)
val = tf.data.Dataset.load(val_save_path)
test = tf.data.Dataset.load(test_save_path)

print("Dataset cargado con éxito.")

# Verifico algunos datos del dataset
for images, labels in train.take(1):  # Toma un lote de imágenes y etiquetas
    print(images.shape, labels.shape)  # Muestra las formas de los tensores cargados

# Ejemplo: ver un batch de datos cargados
data_iterator = train.as_numpy_iterator()
batch = data_iterator.next()

# Mostrar algunas imágenes y etiquetas del batch cargado
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(f"Class: {tf.argmax(batch[1][idx]).numpy()}")


# Cuento las categorias que tengo(diferentes productos)
for _, labels in train.take(1):  # Toma un lote de etiquetas
    cant_categorias = labels.shape[
        -1
    ]  # Toma la dimensión de la última parte de la etiqueta
    break
print("Las categorias son: " + str(cant_categorias))

import numpy as np

# Deep Model-------------------------------------------------------------------
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential

# create augmentation sequentions
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(256, 256, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Definir el modelo
model = Sequential(
    [
        data_augmentation,
        Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(cant_categorias, activation="softmax"),  # 13 clases
    ]
)

# Inicializar y compilar el modelo con CategoricalCrossentropy
CC_loss = CategoricalCrossentropy()
model.compile(optimizer="adam", loss=CC_loss, metrics=["accuracy"])

# Crear un callback para TensorBoard
logdir = "logs"
tensorboard_callback = TensorBoard(log_dir=logdir)

# Calcular steps_per_epoch y validation_steps
# steps_per_epoch = train_size // 16
# validation_steps = val_size // 16

# Entrenar el modelo
history = model.fit(
    train, epochs=100, validation_data=val, callbacks=[tensorboard_callback]
)

# Evaluar el modelo en el conjunto de validación
val_loss, val_accuracy = model.evaluate(val)
print()
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# GUARDO ---------------------------------------------------------------
# Modelo
import datetime

timestamp = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)  # Obtener el timestamp actual
# O guardar en formato SavedModel (formato por defecto de TensorFlow)
save_path_tf = os.path.join(
    "..", "..", "src", "models", f"saved_model_{timestamp}.keras"
)

model.save(save_path_tf)

# Historial
import pickle

save_path_pkl = os.path.join(
    "..", "..", "experiments", "experiment_100iter", f"saved_history"
)
os.makedirs(save_path_pkl, exist_ok=True)

# Guardar el objeto history
with open(os.path.join(save_path_pkl, f"history_{timestamp}.pkl"), "wb") as file_pi:
    pickle.dump(history.history, file_pi)
