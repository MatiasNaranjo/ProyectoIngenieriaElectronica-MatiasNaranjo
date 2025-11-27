from pathlib import Path

import torch
import torchvision
from ultralytics import YOLO


def entrenar_yolo(
    base_path=None,
    version=8,
    imgsz=1440,
    batch=2,
    epochs=300,
    device=0,
    amp=False,
    workers=0,
):
    """
    Entrena un modelo YOLO en un dataset descargado desde Roboflow.

    Parámetros:
        base_path (str | Path): ruta base donde se encuentra data/yolo
        version (int): versión del dataset
        imgsz (int): tamaño de la imagen
        batch (int): tamaño de batch
        epochs (int): número de epochs
        device (int | str): GPU a usar ('0' o 'cpu')
        amp (bool): si usar mixed precision
        workers (int): número de workers para dataloader

    Returns:
        model: objeto YOLO entrenado
    """
    # Verifico si PyTorch, CUDA y torchvision están instalados correctamente
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    print(torchvision.__version__)

    # Cargo el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")

    # Defino la ruta al archivo data.yaml del dataset
    base_path = Path(base_path) if base_path else Path.cwd()
    yaml_path = base_path / f"data/yolo/Proyecto_final_electronica-{version}/data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML en {yaml_path}")

    # Configuro y comienzo el entrenamiento del modelo YOLO
    model.train(
        data=yaml_path,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        workers=workers,
        device=device,
        verbose=True,
        amp=amp,
    )

    return model
