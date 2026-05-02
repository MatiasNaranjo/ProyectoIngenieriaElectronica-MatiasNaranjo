from pydantic import BaseModel, Field


class RaspberryConfig(BaseModel):
    dir_base: str
    dir_config: str | None = None
    dir_utils: str | None = None
    dir_utils_config: str | None = None
    dir_types: str | None = None
    user: str
    ip: str
    folders_down: list[str] = Field(default_factory=list)


class PcConfig(BaseModel):
    dir_base: str
    dir_raspi: str | None = None
    dir_config: str | None = None
    dir_types: str | None = None
    dir_utils: str | None = None
    file_config_loader: list[str] = Field(default_factory=list)
    key_path: str | None = None
    passphrase: str | None = None
    files_up: list[str] = Field(default_factory=list)
    folders_up: list[str] = Field(default_factory=list)
    prefix_config: str | None = None
    prefix_types: str | None = None


class AppConfig(BaseModel):
    raspi: RaspberryConfig
    pc: PcConfig
