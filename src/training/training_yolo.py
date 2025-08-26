import torch
import torchvision
from ultralytics import YOLO


def entrenar_yolo(abs_path, version):
    # Verifico si PyTorch, CUDA y torchvision están instalados correctamente
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    print(torchvision.__version__)

    # Cargo el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")
    yaml_path = (
        abs_path
        + "/data/yolo/Proyecto_final_electronica-"
        + str(version)
        + "/data.yaml"
    )

    # Configuro y comienzo el entrenamiento del modelo YOLO
    model.train(
        data=yaml_path,
        imgsz=1080,
        batch=2,
        epochs=4,
        workers=0,
        device=0,
        verbose=True,
        amp=False,  # Desactiva AMP explícitamente
        # project="resultados_yolo",  # Carpeta donde se guardará el modelo
    )
    return model
