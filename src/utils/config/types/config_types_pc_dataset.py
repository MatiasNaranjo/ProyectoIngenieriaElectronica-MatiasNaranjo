from pydantic import BaseModel


class PipelineConfig(BaseModel):
    download: bool = True


class RoboflowConfig(BaseModel):
    api_key: str
    version: int | None = None
    workspace: str | None = None
    project_name: str | None = None
    yolo_ver: str | None = None


class PathsConfig(BaseModel):
    data_yolo: str | None = None
    data_yolo_raw: str | None = None


class AppConfig(BaseModel):
    pipeline: PipelineConfig
    roboflow: RoboflowConfig
    paths: PathsConfig
