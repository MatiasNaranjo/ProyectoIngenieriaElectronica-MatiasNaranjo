import os

from roboflow import Roboflow


def descargar_dataset(abs_path, version=2, api_key="TU_API_KEY", yolo_ver="yolov8"):
    # Definir la ruta donde se descargarán los datos del modelo
    download_dir = os.path.join(abs_path, "data", "yolo")
    os.makedirs(download_dir, exist_ok=True)
    os.chdir(download_dir)

    # Conectarse a Roboflow y descargar el dataset
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("proyecto-final-labels").project(
        "proyecto_final_electronica"
    )
    dataset = project.version(version).download(yolo_ver)

    # Volver al directorio principal
    os.chdir(abs_path)
    return dataset
