import importlib
import platform
from dataclasses import is_dataclass
from pathlib import Path

import yaml
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parents[2]


class ConfigLoader:
    def __init__(self, func_name: str):
        """
        Loader centralizado de configuración para el proyecto.

        Se encarga de:
        - Cargar y combinar configuraciones desde archivos YAML y variables de entorno.
        - Validar la configuración final mediante modelos Pydantic.
        - Determinar el directorio base del proyecto.
        - Detectar el dispositivo de ejecución (CPU / GPU).

        Este loader permite definir valores por defecto en YAML y sobrescribirlos
        de forma segura mediante variables de entorno, manteniendo una estructura
        tipada y validada.

        Atributos
        ----------
        func_name: str
            Nombre de la funcionalidad
        device: str
            Dispositivo de ejecución
        """
        self.func_name = func_name
        self.device = self.detect_device()

    def _load_config_class(self):
        # Construye dinámicamente el path del módulo de configuración
        module_path = (
            f"src.utils.config.types.config_types_{self.device}_{self.func_name}"
        )

        # Importa el módulo de configuración en tiempo de ejecución
        # Ejemplo: src.utils.config.types.config_types_pc_training
        module = importlib.import_module(module_path)

        class_name = "AppConfig"
        # Obtiene la clase AppConfig desde el módulo importado
        config_class = getattr(module, class_name)

        if not is_dataclass(config_class):
            raise TypeError(f"{class_name} tiene que se un dataclass")

        # Devuelve la clase de configuración (no la instancia)
        return config_class

    def _load_yaml(self):
        """
        Carga la configuración desde un archivo YAML específico según
        el dispositivo y la funcionalidad.

        El nombre del archivo sigue el patrón:
        config_deploy_<device>_<func_name>.yaml

        Si el archivo no existe, retorna un diccionario vacío para permitir
        el uso de valores por defecto o variables de entorno.
        """
        # Nombre dinámico del archivo de configuración
        filename = f"config_deploy_{self.device}_{self.func_name}.yaml"

        # Ruta completa al archivo YAML dentro del proyecto
        path = BASE_DIR / "config" / filename

        # Si el archivo no existe, no se rompe el flujo
        if not path.exists():
            return {}

        # Cargar YAML de forma segura
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_env(self):
        """
        Carga variables de entorno desde un archivo .env específico según
        el dispositivo y la funcionalidad.

        El nombre del archivo sigue el patrón:
        <device>_<func_name>.env

        Si el archivo no existe, retorna un diccionario vacío.
        """
        # Nombre dinámico del archivo .env
        filename = f"{self.device}_{self.func_name}.env"

        # Ruta completa al archivo .env
        path = BASE_DIR / "env" / filename

        # Si no existe el archivo, no se interrumpe el flujo
        if not path.exists():
            return {}

        # Cargar variables de entorno desde el archivo
        return dotenv_values(path)

    def detect_device(self):
        """
        Detecta si se está ejecutando en:
        - PC Windows → 'pc'
        - Raspberry Pi o Linux ARM → 'raspi'
        """
        return "raspi" if platform.system().lower() != "windows" else "pc"
