import os
from pathlib import Path

from roboflow import Roboflow


def clean_dataset_names(dataset_folder):
    """
    Limpia nombres de imágenes y etiquetas de un dataset Roboflow:
    - Toma solo la parte del nombre antes de '_jpg'
    - Elimina '.rf.<hash>'
    """
    # Crea Path del dataset
    dataset_folder = Path(dataset_folder)

    # Recorrer todo el árbol de directorios
    for subfolder in ["images", "labels"]:
        for folder in dataset_folder.rglob(subfolder):
            for file in folder.glob("*.*"):
                # Tomar solo la parte a la izquierda de '_jpg'
                base_name = file.stem.split("_jpg")[0]

                # Forma el nuevo nombre
                new_name = base_name + file.suffix
                file.rename(file.with_name(new_name))


def descargar_dataset(
    version=2,
    api_key=None,
    yolo_ver="yolov8",
    base_path=None,
    data_dir="data/yolo",
    workspace="proyecto-final-labels",
    project_name="proyecto_final_electronica",
):
    """
    Descarga un dataset de Roboflow en formato YOLO y lo guarda en base_path/data/yolo/
        Parámetros:
        version (int): versión del dataset en Roboflow.
        api_key (str): clave de API de Roboflow (requerida).
        yolo_ver (str): formato YOLO a descargar (ej: 'yolov8').
        base_path (str | Path): ruta base donde crear /data/yolo. Si es None -> cwd.
        workspace (str): nombre del workspace de Roboflow.
        project_name (str): nombre del proyecto en Roboflow.

    Returns:
        dataset: objeto Dataset descargado desde Roboflow.
    """
    if not api_key:
        raise ValueError("Debes proporcionar una API key de Roboflow.")

    # Guardar directorio actual
    cwd_original = Path.cwd()

    # Si no especifica un directorio, utiliza el directorio actual
    base_path = Path(base_path) if base_path else cwd_original

    # Definir la ruta donde se descargarán los datos del modelo
    download_dir = base_path / data_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Cambiar a carpeta destino porque Roboflow lo necesita
        os.chdir(download_dir)

        # Inicializar Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)

        # Descargar el dataset
        dataset = project.version(version).download(yolo_ver)

        # Limpiar nombres de archivos descargados
        clean_dataset_names(dataset.location)

    finally:
        # Volver al directorio principal
        os.chdir(cwd_original)

    return dataset
