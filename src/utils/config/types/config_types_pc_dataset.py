from pydantic import BaseModel


class PipelineConfig(BaseModel):
    download: bool = True
    merge: bool = True
    split: bool = True


class RoboflowConfig(BaseModel):
    api_key: str
    version: int | None = None
    workspace: str | None = None
    project_name: str | None = None
    yolo_ver: str | None = None


class DatasetSplitConfig(BaseModel):
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1


class DatasetConfig(BaseModel):
    split_cfg: DatasetSplitConfig
    seed: int = 42


class PathsConfig(BaseModel):
    data_yolo: str | None = None
    data_yolo_raw: str | None = None
    data_yolo_merged: str | None = None


class AppConfig(BaseModel):
    pipeline: PipelineConfig
    roboflow: RoboflowConfig
    dataset: DatasetConfig
    paths: PathsConfig
