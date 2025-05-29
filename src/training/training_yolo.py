import torch
import torchvision
from ultralytics import YOLO

# Verifico si PyTorch, CUDA y torchvision están instalados correctamente
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torchvision.__version__)

# Cargo el modelo YOLO preentrenado
model = YOLO("yolov8n.pt")
ver = 2

# import logging
# logging.basicConfig(level=logging.DEBUG)

yaml_path = (
    "D:/matna/Documents/Escritorio/Facultad/Proyecto/data/yolo/yolo-2/data_mod.yaml"
)


# Configuro y comienzo el entrenamiento del modelo YOLO
model.train(
    data=yaml_path,
    imgsz=640,
    batch=2,
    epochs=400,
    workers=0,
    device=0,
    verbose=True,
    amp=False,  # Desactiva AMP explícitamente
    # project="resultados_yolo",  # Carpeta donde se guardará el modelo
    # name="experimento_1"  # Nombre específico de la subcarpeta
)
