from pathlib import Path

import torch
import torchvision
from ultralytics import YOLO


def entrenar_yolo(
    yaml_path=None,
    version=8,
    imgsz=1440,
    batch=2,
    epochs=300,
    device=0,
    amp=False,
    workers=0,
    cache=False,
    close_mosaic=10,
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
    yaml_path = Path(yaml_path)

    # Verifico si PyTorch, CUDA y torchvision están instalados correctamente
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    print(torchvision.__version__)

    # Cargo el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")

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
        cache=cache,
        close_mosaic=close_mosaic,
    )

    return model
