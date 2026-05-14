from pydantic import BaseModel


class PipelineConfig(BaseModel):
    download: bool = True
    merge: bool = True
    split: bool = True
    experiment: bool = True
    inspect: bool = True


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


class ExperimentConfig(BaseModel):
    train_ratios: list[float] = [0.25, 0.50, 1.0]
    output_dir: str | None = None  # Si es None, se usa data_yolo/experiments
    seed: int = 42


class DatasetConfig(BaseModel):
    split_cfg: DatasetSplitConfig
    seed: int = 42
    mode_sessions: str = "all"  # "all", "common"
    experiment: ExperimentConfig = ExperimentConfig()


class PathsConfig(BaseModel):
    data_yolo: str | None = None
    data_yolo_raw: str | None = None
    data_yolo_merged: str | None = None
    data_yolo_split: str | None = None
    data_experiments: str | None = None
    yaml: str | None = None


class AppConfig(BaseModel):
    pipeline: PipelineConfig
    roboflow: RoboflowConfig
    dataset: DatasetConfig
    paths: PathsConfig
