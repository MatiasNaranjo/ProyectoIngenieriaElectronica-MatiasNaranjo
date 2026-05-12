from src.training.training_yolo import (
    entrenar_experimentos,
    entrenar_yolo,
    summarize_experiments,
)
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("train").load()

    if config.pipeline.train:
        # Entrenar modelo YOLO
        entrenar_yolo(
            yaml_path=config.paths.yaml,
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

    if config.pipeline.experiment:
        # Entrenar experimentos con diferentes ratios de datos
        entrenar_experimentos(
            experiments_dir=config.paths.experiments_dir,
            output_dir=config.paths.experiments_run_dir,
            model=config.experiment.model,
            imgsz=config.experiment.imgsz,
            batch=config.experiment.batch,
            epochs=config.experiment.epochs,
            device=config.experiment.device,
            amp=config.experiment.amp,
            workers=config.experiment.workers,
            cache=config.experiment.cache,
            close_mosaic=config.experiment.close_mosaic,
            cos_lr=config.experiment.cos_lr,
            train_ratios=config.experiment.train_ratios,
        )

    if config.pipeline.summarize:
        # Generar summary de experimentos
        summarize_experiments(
            output_dir=config.paths.experiments_run_dir,
        )


if __name__ == "__main__":
    main()
