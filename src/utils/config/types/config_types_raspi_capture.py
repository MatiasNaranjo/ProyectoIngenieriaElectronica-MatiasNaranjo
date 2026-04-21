from pydantic import BaseModel


class CameraConfig(BaseModel):
    clase: str
    output_dir: str
    n_photos: int | None = None
    resolution: int | None = None
    delay: float | None = None


class AppConfig(BaseModel):
    cam: CameraConfig
