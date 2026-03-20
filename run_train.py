from src.training.training_yolo import entrenar_yolo
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("train").load()

    # Entrenar modelo YOLO
    entrenar_yolo(
        yaml_path=config.paths.yaml,
        version=config.training.version,
        imgsz=config.training.imgsz,
        batch=config.training.batch,
        epochs=config.training.epochs,
        device=config.training.device,
        amp=config.training.amp,
        workers=config.training.workers,
        cache=config.training.cache,
        close_mosaic=config.training.close_mosaic,
        cos_lr=config.training.cos_lr,
    )


if __name__ == "__main__":
    main()
