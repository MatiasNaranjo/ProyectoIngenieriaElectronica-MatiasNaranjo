import glob
import os
import pickle

import matplotlib.pyplot as plt

load_path = os.path.join(
    "..", "..", "experiments", "experiment_2", "saved_history"
)  # Ruta relativa
history_files = glob.glob(os.path.join(load_path, "history_*.pkl"))

# Seleccionar el archivo más reciente
latest_history_file = max(history_files, key=os.path.getctime)

# Cargar el objeto history más reciente
with open(latest_history_file, "rb") as file_pi:
    history = pickle.load(file_pi)

# Grafica
fig = plt.figure()
plt.plot(history["loss"], color="teal", label="loss")
plt.plot(history["val_loss"], color="orange", label="val_loss")
fig.suptitle("Loss", fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Grafica
fig = plt.figure()
plt.plot(history["accuracy"], color="teal", label="accuracy")
plt.plot(history["val_accuracy"], color="orange", label="val_accuracy")
fig.suptitle("Accuracy", fontsize=20)
plt.legend(loc="upper left")
plt.show()
