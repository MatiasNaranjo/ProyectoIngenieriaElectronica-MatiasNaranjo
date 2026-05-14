from pydantic import BaseModel


class PipelineConfig(BaseModel):
    train: bool | None = None
    experiment: bool | None = None
    summarize: bool | None = None


class TrainingConfig(BaseModel):
    version: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    epochs: int | None = None
    device: int | None = None
    amp: bool | None = None
    workers: int | None = None
    cache: str | None = None
    close_mosaic: int | None = None
    cos_lr: bool | None = None


class ExperimentConfig(BaseModel):
    version: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    epochs: int | None = None
    device: int | None = None
    amp: bool | None = None
    workers: int | None = None
    cache: str | None = None
    close_mosaic: int | None = None
    cos_lr: bool | None = None
    model: str | None = None
    train_ratios: list[float] | None = None


class PathConfig(BaseModel):
    yaml: str | None = None
    experiments_run_dir: str | None = None
    experiments_dir: str | None = None


class AppConfig(BaseModel):
    pipeline: PipelineConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    paths: PathConfig
