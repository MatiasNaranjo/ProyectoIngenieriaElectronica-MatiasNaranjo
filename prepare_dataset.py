from src.data.data_yolo import descargar_dataset
from src.data.inspect_dataset import DatasetInspector
from src.data.merge_dataset import merge_roboflow_dataset
from src.data.split_dataset import split_by_session, split_experiment
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("dataset").load()

    # Descargar dataset desde Roboflow
    if config.pipeline.download:
        descargar_dataset(
            version=config.roboflow.version,
            api_key=config.roboflow.api_key,
            yolo_ver=config.roboflow.yolo_ver,
            data_dir=config.paths.data_yolo_raw,
            workspace=config.roboflow.workspace,
            project_name=config.roboflow.project_name,
        )

    # Mergear los splits del dataset descargado en una sola carpeta
    if config.pipeline.merge:
        merge_roboflow_dataset(
            input_dir=config.paths.data_yolo_raw,
            output_dir=config.paths.data_yolo,
        )

    # Dividir el dataset en splits de sesiones
    if config.pipeline.split:
        split_by_session(
            input_dataset_path=config.paths.data_yolo_merged,
            output_dataset_path=config.paths.data_yolo,
            yaml_path=config.paths.yaml,
            split_ratio=config.dataset.split_cfg.model_dump(),
            seed=config.dataset.seed,
            mode_sessions=config.dataset.mode_sessions,
        )

        # Generar subsets de entrenamiento para análisis de sensibilidad
    if config.pipeline.experiment:
        split_experiment(
            split_path=config.paths.data_yolo_split,
            output_path=config.paths.data_experiments,
            yaml_path=config.paths.yaml,
            train_ratios=config.dataset.experiment.train_ratios,
            seed=config.dataset.experiment.seed,
        )

    # Inspeccionar el dataset ya preparado
    if config.pipeline.inspect:
        inspector = DatasetInspector(config.paths.data_yolo)
        # Imprime el resumen del dataset
        inspector.summary_split()


if __name__ == "__main__":
    main()
