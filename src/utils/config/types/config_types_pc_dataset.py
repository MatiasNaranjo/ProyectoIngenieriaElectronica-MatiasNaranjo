from pydantic import BaseModel


class RoboflowConfig(BaseModel):
    api_key: str
    version: int | None = None
    workspace: str | None = None
    project_name: str | None = None
    yolo_ver: str | None = None


class PathsConfig(BaseModel):
    data_yolo: str | None = None


class AppConfig(BaseModel):
    roboflow: RoboflowConfig
    paths: PathsConfig
