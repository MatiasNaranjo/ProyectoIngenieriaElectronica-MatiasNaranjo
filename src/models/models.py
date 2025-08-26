import glob
import os
import pickle

import matplotlib.pyplot as plt


def load_latest_history(experimento, base_dir="../../experiments"):
    """
    Carga el archivo de history más reciente de un experimento.
    """
    load_path = os.path.join(base_dir, experimento, "saved_history")
    history_files = glob.glob(os.path.join(load_path, "history_*.pkl"))
    if not history_files:
        raise FileNotFoundError(f"No se encontraron archivos en {load_path}")

    latest_history_file = max(history_files, key=os.path.getctime)
    with open(latest_history_file, "rb") as file_pi:
        history = pickle.load(file_pi)

    print(f"History cargado desde {latest_history_file}")
    return history


def plot_and_save(history, experimento, metric, save_base="../../reports/figures"):
    """
    Genera y guarda una gráfica para una métrica y su validación.
    """
    fig = plt.figure()
    plt.plot(history[metric], color="teal", label=metric)
    plt.plot(history[f"val_{metric}"], color="orange", label=f"val_{metric}")
    fig.suptitle(metric.capitalize(), fontsize=20)
    plt.legend(loc="upper left")

    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, f"{experimento}_{metric}.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Figura de {metric} guardada en {save_path}")


def plot_history_metrics(experimento, metrics=("loss", "accuracy")):
    """
    Carga el history más reciente y genera las gráficas para las métricas indicadas.
    """
    history = load_latest_history(experimento)
    for metric in metrics:
        if metric in history and f"val_{metric}" in history:
            plot_and_save(history, experimento, metric)
        else:
            print(f"La métrica {metric} no está en el history")
