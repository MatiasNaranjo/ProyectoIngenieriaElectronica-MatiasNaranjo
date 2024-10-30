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

# Split Data
total_size = len(data)

train_size = int(total_size * 0.7)
val_size = int(total_size * 0.2)
test_size = total_size - train_size - val_size

print("Los batches de entrenamiento son: " + str(train_size))
print("Los batches de validación son: " + str(val_size))
print("Los batches de prueba son: " + str(test_size))
print("Los batches totales son: " + str(total_size))

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


# Guardar---------------------------------------------------------
# Directorio base donde guardaremos los conjuntos
base_save_path = os.path.join("..", "..", "data", "processed")

# Guardar cada conjunto por separado
train_save_path = os.path.join(base_save_path, "train")
val_save_path = os.path.join(base_save_path, "val")
test_save_path = os.path.join(base_save_path, "test")

# Crear directorios si no existen
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(val_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)

# Guardar los datasets
tf.data.Dataset.save(train, train_save_path)
tf.data.Dataset.save(val, val_save_path)
tf.data.Dataset.save(test, test_save_path)

print(f"Tensores guardados en {base_save_path}")
