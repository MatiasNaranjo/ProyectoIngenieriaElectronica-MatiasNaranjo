import os
import subprocess

import tensorflow as tf

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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, Sequential


# create augmentation sequentions
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(256, 256, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Definir el modelo


# Configuración de data augmentation
data_augmentation = Sequential(
    [
        # Ejemplo de aumentación de datos; puedes agregar más según sea necesario.
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

# Cargar ResNet50 como base, sin las capas superiores
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

# Congelar las capas de ResNet para mantener los pesos preentrenados
for layer in base_model.layers:
    layer.trainable = False

# Definir el modelo en forma secuencial con capas adicionales
model = Sequential(
    [
        data_augmentation,  # Aumentación de datos
        base_model,  # Base preentrenada de ResNet
        MaxPooling2D(),  # Capa de pooling adicional
        Conv2D(128, (3, 3), activation="relu"),  # Capas adicionales de convolución
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(cant_categorias, activation="softmax"),  # Salida con tus clases
    ]
)

# Compilar el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Crear un callback para TensorBoard
logdir = "logs"
tensorboard_callback = TensorBoard(log_dir=logdir)

# Entrenar el modelo
history = model.fit(
    train, epochs=100, validation_data=val, callbacks=[tensorboard_callback]
)

# Inicializar y compilar el modelo con CategoricalCrossentropy
CC_loss = CategoricalCrossentropy()
model.compile(optimizer="adam", loss=CC_loss, metrics=["accuracy"])


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
    "..", "..", "src", "models", f"saved_modelRest_{timestamp}.keras"
)

model.save(save_path_tf)

# Historial
import pickle

save_path_pkl = os.path.join(
    "..", "..", "experiments", "experiment_Rest", f"saved_history"
)
os.makedirs(save_path_pkl, exist_ok=True)

# Guardar el objeto history
with open(os.path.join(save_path_pkl, f"history_{timestamp}.pkl"), "wb") as file_pi:
    pickle.dump(history.history, file_pi)
