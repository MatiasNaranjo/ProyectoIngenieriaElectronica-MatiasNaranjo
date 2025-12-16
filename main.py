from src.data.data_yolo import descargar_dataset
from src.training.training_yolo import entrenar_yolo
from src.utils.config_loader import BASE_DIR, ConfigLoader


def main():
    config = ConfigLoader("train").load()
    # Descargar dataset desde Roboflow
    descargar_dataset(
        version=config.roboflow.version,
        api_key=config.roboflow.api_key,
        yolo_ver=config.roboflow.yolo_ver,
        base_path=BASE_DIR,
        workspace=config.roboflow.workspace,
        project_name=config.roboflow.project_name,
    )

    # Entrenar modelo YOLO
    entrenar_yolo(
        base_path=BASE_DIR,
        version=config.training.version,
        imgsz=config.training.imgsz,
        batch=config.training.batch,
        epochs=config.training.epochs,
        device=config.training.device,
        amp=config.training.amp,
        workers=config.training.workers,
    )


if __name__ == "__main__":
    main()
