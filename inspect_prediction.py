from src.evaluations.prediction_analizer import PredictionAnalyzer
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("inspection").load()

    inspector = PredictionAnalyzer(
        model_path=config.path.model,
        dataset_path=config.path.data_yolo,
        iou_threshold=config.inspect.iou_threshold,
        conf_threshold=config.inspect.conf_threshold,
    )

    inspector.run(
        splits=config.inspect.splits,
        output_dir=config.inspect.output_dir,
        save_images=config.inspect.save_images,
    )


if __name__ == "__main__":
    main()
