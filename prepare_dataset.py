from src.data.data_yolo import descargar_dataset
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("dataset").load()

    # Descargar dataset desde Roboflow
    descargar_dataset(
        version=config.roboflow.version,
        api_key=config.roboflow.api_key,
        yolo_ver=config.roboflow.yolo_ver,
        data_dir=config.paths.data_yolo,
        workspace=config.roboflow.workspace,
        project_name=config.roboflow.project_name,
    )


if __name__ == "__main__":
    main()
