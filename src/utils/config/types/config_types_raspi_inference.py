from pydantic import BaseModel


class InferenceConfig(BaseModel):
    frame_delay: float | None = None
    inference_dir: str
    max_frames: int | None = None
    model_path: str
    resolution: int | None = None


class AppConfig(BaseModel):
    inference: InferenceConfig
