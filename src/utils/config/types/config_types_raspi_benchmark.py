from pydantic import BaseModel


class BenchmarkConfig(BaseModel):
    model_path: str
    resolution: int | None = None
    conf: float = 0.5
    iou: float = 0.3
    num_frames: int = 200
    warmup: int = 10
    output_csv: str


class AppConfig(BaseModel):
    benchmark: BenchmarkConfig
