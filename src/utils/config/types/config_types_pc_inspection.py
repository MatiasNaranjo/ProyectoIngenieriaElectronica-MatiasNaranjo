from pydantic import BaseModel


class InspectConfig(BaseModel):
    iou_threshold: float = 0.5
    conf_threshold: float = 0.5
    splits: list[str] = ["val", "test"]
    output_dir: str = "reports/errors"
    save_images: bool = True


class PathsConfig(BaseModel):
    data_yolo: str
    model: str


class AppConfig(BaseModel):
    inspect: InspectConfig
    path: PathsConfig
