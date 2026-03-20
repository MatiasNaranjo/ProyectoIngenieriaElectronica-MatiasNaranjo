from pydantic import BaseModel


class TrainingConfig(BaseModel):
    version: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    epochs: int | None = None
    device: int | None = None
    amp: bool | None = None
    workers: int | None = None
    cache: str | None = None


class PathConfig(BaseModel):
    yaml: str | None = None


class AppConfig(BaseModel):
    training: TrainingConfig
    paths: PathConfig
