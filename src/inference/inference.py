import glob
import os
import pickle

import matplotlib.pyplot as plt

experimento = "experiment_100iter"
load_path = os.path.join(
    "..", "..", "experiments", experimento, "saved_history"
)  # Ruta relativa
history_files = glob.glob(os.path.join(load_path, "history_*.pkl"))

# Seleccionar el archivo más reciente
latest_history_file = max(history_files, key=os.path.getctime)

# Cargar el objeto history más reciente
with open(latest_history_file, "rb") as file_pi:
    history = pickle.load(file_pi)

# Grafica Loss
fig = plt.figure()
plt.plot(history["loss"], color="teal", label="loss")
plt.plot(history["val_loss"], color="orange", label="val_loss")
fig.suptitle("Loss", fontsize=20)
plt.legend(loc="upper left")
# Crea la ruta y asegura de que exista
save_dir = os.path.join("..", "..", "reports", "figures")
os.makedirs(save_dir, exist_ok=True)  # Crea la carpeta si no existe
# Guardar la figura en la ruta especificada
save_path = os.path.join(save_dir, experimento + "_" + "loss.png")
plt.savefig(save_path)  # Ajustar dpi si necesitas mayor resolución

print(f"Figura guardada en {save_path}")

# Grafica Accuracy
fig = plt.figure()
plt.plot(history["accuracy"], color="teal", label="accuracy")
plt.plot(history["val_accuracy"], color="orange", label="val_accuracy")
fig.suptitle("Accuracy", fontsize=20)
plt.legend(loc="upper left")
# Guardar la figura en la ruta especificada
save_path = os.path.join(save_dir, experimento + "_" + "accuracy.png")
plt.savefig(save_path)  # Ajustar dpi si necesitas mayor resolución

print(f"Figura guardada en {save_path}")
