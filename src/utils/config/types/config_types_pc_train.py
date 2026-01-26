from pydantic import BaseModel


class RoboflowConfig(BaseModel):
    api_key: str | None = None
    version: int | None = None
    workspace: str | None = None
    project_name: str | None = None
    yolo_ver: str | None = None


class TrainingConfig(BaseModel):
    version: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    epochs: int | None = None
    device: int | None = None
    amp: bool | None = None
    workers: int | None = None


class AppConfig(BaseModel):
    roboflow: RoboflowConfig
    training: TrainingConfig
