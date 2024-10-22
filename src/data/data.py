import imghdr
import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

data_dir = r"D:\matna\Documents\Escritorio\Facultad\Proyecto\data\raw\data"
categorias = os.listdir(data_dir)

cant_categorias = len(categorias)
print("Las " + str(cant_categorias) + " categorias son: " + str(categorias))

# Contar archivos en cada categoría
for categoria in categorias:
    categoria_path = os.path.join(data_dir, categoria)
    if os.path.isdir(categoria_path):
        num_archivos = len(os.listdir(categoria_path))
        print(f"- {num_archivos} fotos de {categoria}")


# Cargar el dataset de imágenes
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(256, 256),
    batch_size=16,
    label_mode="int",  # Asegúrate de que las etiquetas sean enteros
)

data_iterator = data.as_numpy_iterator()

# Get another batch from the iterator
batch = data_iterator.next()

print()
print(batch[0].shape)
print(batch[1])

# Muestro 4 imagenes
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# Normalizar las imágenes y convertir etiquetas a one-hot encoding
def process_data(image, label):
    image = image / 255.0
    label = tf.one_hot(
        label, cant_categorias
    )  # Convertir las etiquetas a one-hot encoding
    return image, label


data = data.map(process_data)


scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)

    # Convertir la etiqueta one-hot a la clase original
    label_class = tf.argmax(batch[1][idx]).numpy()

    ax[idx].title.set_text(f"Class: {label_class}")

# Guardar---------------------------------------------------------
# Guardar el tensor usando TensorFlow
save_path = os.path.join("..", "..", "data", "processed")  # Ruta relativa
tf.data.experimental.save(data, save_path)
print(f"Tensores guardados en {save_path}")
# print(os.getcwd())  # Te muestra el directorio actual
