from pydantic import BaseModel, Field


class RaspberryConfig(BaseModel):
    dir: str
    user: str
    ip: str
    folders_down: list[str] = Field(default_factory=list)


class PcConfig(BaseModel):
    dir: str
    key_path: str | None = None
    passphrase: str | None = None
    files_up: list[str] = Field(default_factory=list)
    folders_up: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    raspi: RaspberryConfig
    pc: PcConfig
