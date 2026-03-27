from pathlib import Path

from ultralytics import YOLO


class ErrorInspector:
    """
    Analiza los errores de un modelo YOLO sobre un dataset spliteado.
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
